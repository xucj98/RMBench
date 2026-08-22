#!/usr/bin/env python3
"""Convert tagged X1Pro drawer-sorting demonstrations to shared-memory LeRobot data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import copy
import dataclasses
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}
STATE_KEYS = (
    "follow_left_position",
    "follow_left_rotation",
    "follow_left_gripper",
    "follow_right_position",
    "follow_right_rotation",
    "follow_right_gripper",
    "master_left_position",
    "master_left_rotation",
    "master_left_gripper",
    "master_right_position",
    "master_right_rotation",
    "master_right_gripper",
)


@dataclasses.dataclass(frozen=True)
class Interval:
    start: int
    end: int
    annotation_label: str
    item_id: int
    kind: str


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    total_frames: int
    source_fps: float
    intervals: tuple[Interval, ...]
    missing_observation_items: tuple[int, ...]
    annotation_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_repo_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    schema_ref = config.get("memory_schema")
    if not schema_ref:
        raise ValueError("Converter config must define memory_schema")
    schema_path = (config_path.parent / str(schema_ref)).resolve()
    schema = _load_yaml(schema_path)
    fields = schema.get("fields", [])
    if len(fields) != 1 or fields[0].get("name") != "drawer_target":
        raise ValueError("Drawer sorting requires exactly one drawer_target memory field")
    resolved = copy.deepcopy(config)
    resolved["memory"] = copy.deepcopy(fields[0])
    resolved.setdefault("_runtime", {}).update(
        {"config_path": str(config_path.resolve()), "memory_schema_path": str(schema_path)}
    )
    return resolved


def _episode_json_path(episode_path: Path) -> Path:
    return episode_path / f"{episode_path.name}.json"


def _episode_metadata(path: Path) -> tuple[int, float]:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.read(4096)
    total_match = re.search(r'"total"\s*:\s*(\d+)', header)
    fps_match = re.search(r'"fps"\s*:\s*([0-9.]+)', header)
    if total_match is None or fps_match is None:
        payload = _load_json(path)
        total = int(payload.get("total", len(payload["data"])))
        fps = float(payload.get("fps", 30.0))
    else:
        total = int(total_match.group(1))
        fps = float(fps_match.group(1))
    if total < 2 or fps <= 0:
        raise ValueError(f"Invalid episode metadata total={total}, fps={fps}")
    return total, fps


def _annotation_mapping(memory: dict[str, Any]) -> tuple[dict[str, int], dict[str, int]]:
    labels = list(memory["labels"])
    annotation = memory["annotation"]

    def convert(mapping: dict[str, str]) -> dict[str, int]:
        result = {}
        for annotation_label, value in mapping.items():
            if value not in labels:
                raise ValueError(f"Memory value {value!r} is not in labels {labels}")
            result[str(annotation_label)] = labels.index(value)
        return result

    return convert(annotation["observe_labels"]), convert(annotation["execution_labels"])


def _load_intervals(
    annotation_path: Path,
    total_frames: int,
    memory: dict[str, Any],
) -> tuple[tuple[Interval, ...], tuple[int, ...]]:
    payload = _load_json(annotation_path)
    observe_mapping, execution_mapping = _annotation_mapping(memory)
    intervals = []
    observed_items = set()
    execution_items = set()

    for kind, mapping in (("observe", observe_mapping), ("execute", execution_mapping)):
        for annotation_label, item_id in mapping.items():
            values = payload.get(annotation_label, [])
            if not isinstance(values, list) or len(values) % 2 != 0:
                raise ValueError(f"Label {annotation_label} must contain start/end pairs, got {values!r}")
            for index in range(0, len(values), 2):
                start = int(round(float(values[index])))
                end = int(round(float(values[index + 1])))
                if not 0 <= start < end <= total_frames:
                    raise ValueError(
                        f"Label {annotation_label} interval [{start}, {end}) is outside [0, {total_frames})"
                    )
                intervals.append(Interval(start, end, annotation_label, item_id, kind))
                (observed_items if kind == "observe" else execution_items).add(item_id)

    intervals.sort(key=lambda item: (item.start, item.end, item.annotation_label))
    if any(current.start < previous.end for previous, current in itertools.pairwise(intervals)):
        raise ValueError(f"Annotation intervals overlap: {intervals}")
    expected_execution_items = set(execution_mapping.values())
    if execution_items != expected_execution_items:
        raise ValueError(
            f"Expected execution data for items {sorted(expected_execution_items)}, got {sorted(execution_items)}"
        )
    missing_observations = tuple(sorted(execution_items - observed_items))
    return tuple(intervals), missing_observations


def discover_episodes(config: dict[str, Any]) -> tuple[list[EpisodeRecord], list[dict[str, str]]]:
    dataset_config = config["dataset"]
    dataset_root = _resolve_repo_path(dataset_config["source_dir"])
    required_tag = str(dataset_config["required_tag"])
    annotation_relative_path = Path(config["memory"]["annotation"]["relative_path"])
    expected_source_fps = float(dataset_config["source_fps"])
    records = []
    skipped = []

    for episode_path in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        tags_path = episode_path / "anno" / "tags.json"
        if not tags_path.is_file():
            continue
        try:
            tags = _load_json(tags_path).get("tags", [])
        except Exception as exc:
            skipped.append({"episode": episode_path.name, "reason": f"invalid tags: {exc}"})
            continue
        if required_tag not in tags:
            continue

        annotation_path = episode_path / annotation_relative_path
        trajectory_path = _episode_json_path(episode_path)
        missing_files = [
            name
            for name in (*FILE_CAMERA_MAPPING.values(), trajectory_path.name, str(annotation_relative_path))
            if not (episode_path / name).is_file()
        ]
        if missing_files:
            skipped.append({"episode": episode_path.name, "reason": f"missing files: {missing_files}"})
            continue
        try:
            total_frames, source_fps = _episode_metadata(trajectory_path)
            if not np.isclose(source_fps, expected_source_fps):
                raise ValueError(f"expected {expected_source_fps} Hz, got {source_fps} Hz")
            intervals, missing_observations = _load_intervals(annotation_path, total_frames, config["memory"])
        except Exception as exc:
            skipped.append({"episode": episode_path.name, "reason": str(exc)})
            continue
        records.append(
            EpisodeRecord(
                path=episode_path,
                total_frames=total_frames,
                source_fps=source_fps,
                intervals=intervals,
                missing_observation_items=missing_observations,
                annotation_path=annotation_path,
            )
        )
    return records, skipped


def semantic_memory_timeline(total_frames: int, intervals: tuple[Interval, ...]) -> np.ndarray:
    """Resolve the latent memory after each source observation.

    Observation starts acquire an item. Execution starts also set the matching
    item so demonstrations with missing observation remain valid conditioned
    action samples. Execution ends return to ``observe``.
    """
    starts: dict[int, list[Interval]] = {}
    ends: dict[int, list[Interval]] = {}
    for interval in intervals:
        starts.setdefault(interval.start, []).append(interval)
        if interval.kind == "execute":
            ends.setdefault(interval.end, []).append(interval)

    result = np.zeros(total_frames, dtype=np.int64)
    current = 0
    for frame_index in range(total_frames):
        for interval in ends.get(frame_index, []):
            if current == interval.item_id:
                current = 0
        for interval in starts.get(frame_index, []):
            current = interval.item_id
        result[frame_index] = current
    return result


def memory_action_valid_timeline(total_frames: int, intervals: tuple[Interval, ...]) -> np.ndarray:
    """Mask dense-memory targets created only by execution-time conditioning.

    An execution interval may supply a known item even when the demonstration
    has no preceding observation of that item. That override is valid as an
    action condition, but it is not evidence for an ``observe -> item`` memory
    transition. We therefore suppress dense-memory action loss for the whole
    forced interval while retaining robot-action supervision.
    """
    starts: dict[int, list[Interval]] = {}
    ends: dict[int, list[Interval]] = {}
    for interval in intervals:
        starts.setdefault(interval.start, []).append(interval)
        if interval.kind == "execute":
            ends.setdefault(interval.end, []).append(interval)

    valid = np.ones(total_frames, dtype=np.bool_)
    current = 0
    for frame_index in range(total_frames):
        for interval in ends.get(frame_index, []):
            if current == interval.item_id:
                current = 0
        for interval in starts.get(frame_index, []):
            if interval.kind == "execute" and current != interval.item_id:
                valid[interval.start : interval.end] = False
            current = interval.item_id
    return valid


def source_indices_for_target_frames(
    target_frame_count: int,
    source_frame_count: int,
    source_fps: float,
    target_fps: int,
) -> np.ndarray:
    target_indices = np.arange(target_frame_count, dtype=np.float64)
    source_indices = np.rint(target_indices * source_fps / target_fps).astype(np.int64)
    return np.clip(source_indices, 0, source_frame_count - 1)


def memory_supervision_for_target_frames(
    record: EpisodeRecord,
    target_frame_count: int,
    target_fps: int,
    query_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_indices = source_indices_for_target_frames(
        target_frame_count, record.total_frames, record.source_fps, target_fps
    )
    source_semantic = semantic_memory_timeline(record.total_frames, record.intervals)
    source_memory_action_valid = memory_action_valid_timeline(record.total_frames, record.intervals)
    target_ids = source_semantic[source_indices]
    memory_action_valid = source_memory_action_valid[source_indices]
    previous_indices = np.maximum(np.arange(target_frame_count) - query_stride, 0)
    input_ids = target_ids[previous_indices].copy()

    # Critical real-data rule: 4/5/6 are actions under known item memory even
    # when their corresponding 1/2/3 observation interval is absent.
    for interval in record.intervals:
        if interval.kind != "execute":
            continue
        target_mask = (source_indices >= interval.start) & (source_indices < interval.end)
        input_ids[target_mask] = interval.item_id
        target_ids[target_mask] = interval.item_id

    target_valid = np.ones((target_frame_count, 1), dtype=np.bool_)
    return input_ids[:, None], target_ids[:, None], target_valid, memory_action_valid[:, None]


def implicit_unknown_one_hot(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if np.any((ids < 0) | (ids > 3)):
        raise ValueError(f"Drawer target ids must be in [0, 3], got {np.unique(ids)}")
    output = np.zeros((len(ids), 3), dtype=np.float32)
    nonzero = ids > 0
    output[np.nonzero(nonzero)[0], ids[nonzero] - 1] = 1.0
    return output


def _load_robot_trajectory(record: EpisodeRecord) -> np.ndarray:
    payload = _load_json(_episode_json_path(record.path))
    data = payload["data"]
    trajectories: dict[str, list[Any]] = {key: [] for key in STATE_KEYS}
    for frame in data:
        for key in STATE_KEYS:
            trajectories[key].append(frame[key])
    arrays = []
    for key in STATE_KEYS:
        value = np.asarray(trajectories[key], dtype=np.float32)
        arrays.append(value.reshape(-1, 1) if "gripper" in key else value)
    robot = np.concatenate(arrays, axis=1)
    if robot.shape != (record.total_frames, 28):
        raise ValueError(f"Expected [{record.total_frames}, 28] SM2SM trajectory, got {robot.shape}")
    return robot


def load_training_arrays(
    record: EpisodeRecord,
    target_frame_count: int,
    target_fps: int,
    query_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    robot_source = _load_robot_trajectory(record)
    source_indices = source_indices_for_target_frames(
        target_frame_count, record.total_frames, record.source_fps, target_fps
    )
    robot = robot_source[source_indices]
    input_ids, target_ids, target_valid, memory_action_valid = memory_supervision_for_target_frames(
        record, target_frame_count, target_fps, query_stride
    )
    dense_input = implicit_unknown_one_hot(input_ids)
    dense_target = implicit_unknown_one_hot(target_ids)
    availability = np.zeros((target_frame_count, 1), dtype=np.float32)
    state = np.concatenate([robot, dense_input, availability], axis=1)
    action = np.concatenate([robot, dense_target, availability], axis=1)
    return state, action, input_ids, target_ids, target_valid, memory_action_valid


def transcode_video_ffmpeg(
    input_path: Path,
    output_path: Path,
    target_size: tuple[int, int],
    fps: int,
    video_codec: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec, crf = ("libx264", 23) if video_codec == "h264" else ("libsvtav1", 30)
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-vf",
        f"scale={target_size[0]}:{target_size[1]}",
        "-c:v",
        codec,
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-g",
        "2",
        "-crf",
        str(crf),
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "SVT_LOG": "0"}, check=False)
    if result.returncode != 0:
        raise RuntimeError("\n".join(result.stderr.splitlines()[-20:]))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {output_path}: {probe.stderr}")
    return int(probe.stdout.strip())


def _dummy_video_stats(num_frames: int) -> dict[str, np.ndarray]:
    return {
        "min": np.zeros((3, 1, 1)),
        "max": np.ones((3, 1, 1)),
        "mean": np.full((3, 1, 1), 0.4),
        "std": np.full((3, 1, 1), 0.25),
        "count": np.asarray([num_frames]),
    }


class NoVideoIOLeRobotDataset(LeRobotDataset):
    """LeRobot dataset whose already-transcoded videos bypass image I/O."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_all_media = False
        self._video_frame_count = 0

    def _save_image(self, image, fpath: Path) -> None:
        if not self._skip_all_media:
            super()._save_image(image, fpath)

    def save_episode(self, episode_data: dict | None = None) -> None:
        if not self._skip_all_media:
            super().save_episode(episode_data)
            return

        from lerobot.common.datasets.compute_stats import get_feature_stats
        from lerobot.common.datasets.lerobot_dataset import aggregate_stats
        from lerobot.common.datasets.lerobot_dataset import validate_episode_buffer
        from lerobot.common.datasets.lerobot_dataset import write_episode
        from lerobot.common.datasets.lerobot_dataset import write_episode_stats
        from lerobot.common.datasets.lerobot_dataset import write_info

        episode_buffer = episode_data or self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]
        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.asarray([self.meta.get_task_index(task) for task in tasks])
        for key, feature in self.features.items():
            if key in {"index", "episode_index", "task_index"} or feature["dtype"] in {"image", "video"}:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])
        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        episode_stats = {}
        for key, value in episode_buffer.items():
            if key not in self.features or self.features[key]["dtype"] == "string":
                continue
            episode_stats[key] = (
                _dummy_video_stats(self._video_frame_count)
                if self.features[key]["dtype"] in {"image", "video"}
                else get_feature_stats(value, axis=0, keepdims=value.ndim == 1)
            )
        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length
        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1
        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        self.meta.info["total_videos"] += len(self.meta.video_keys)
        write_info(self.meta.info, self.meta.root)
        episode_dict = {"episode_index": episode_index, "tasks": episode_tasks, "length": episode_length}
        self.meta.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.meta.root)
        self.meta.episodes_stats[episode_index] = episode_stats
        self.meta.stats = aggregate_stats([self.meta.stats, episode_stats]) if self.meta.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.meta.root)
        if episode_data is None:
            self.episode_buffer = self.create_episode_buffer()

    def finalize_video_info(self) -> None:
        from lerobot.common.datasets.lerobot_dataset import write_info

        self.meta.update_video_info()
        write_info(self.meta.info, self.meta.root)

    @classmethod
    def create(cls, **kwargs) -> NoVideoIOLeRobotDataset:
        parent = LeRobotDataset.create(**kwargs)
        result = cls.__new__(cls)
        result.__dict__.update(parent.__dict__)
        result._skip_all_media = False  # noqa: SLF001
        result._video_frame_count = 0  # noqa: SLF001
        return result


def _create_dataset(config: dict[str, Any], output_path: Path) -> NoVideoIOLeRobotDataset:
    target_size = (320, 240)
    shape = (target_size[1], target_size[0], 3)
    field_names = [field["name"] for field in [config["memory"]]]
    features = {
        **{
            camera: {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"]}
            for camera in FILE_CAMERA_MAPPING
        },
        "state": {"dtype": "float32", "shape": (32,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (32,), "names": ["actions"]},
        "key_state_input_ids": {"dtype": "int64", "shape": (1,), "names": field_names},
        "key_state_target_ids": {"dtype": "int64", "shape": (1,), "names": field_names},
        "key_state_target_mask": {"dtype": "bool", "shape": (1,), "names": field_names},
        "memory_action_valid": {"dtype": "bool", "shape": (1,), "names": ["drawer_target"]},
    }
    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=config["dataset"]["repo_id"],
        root=output_path,
        robot_type="ARX",
        fps=int(config["dataset"]["target_fps"]),
        features=features,
        image_writer_threads=0,
        image_writer_processes=0,
    )
    dataset._skip_all_media = True  # noqa: SLF001
    return dataset


def _write_metadata(
    output_path: Path,
    config: dict[str, Any],
    records: list[EpisodeRecord],
    skipped: list[dict[str, str]],
    argv: list[str],
) -> None:
    memory_dir = output_path / "meta" / "key_state"
    memory_dir.mkdir(parents=True, exist_ok=True)
    phase_layout = {
        "schema_version": 1,
        "encoding": "shared_memory",
        "field": config["memory"],
        "dense_layout": {"robot_dim": 28, "memory_dim": [28, 31], "availability_mask_dim": 31},
        "source_fps": int(config["dataset"]["source_fps"]),
        "target_fps": int(config["dataset"]["target_fps"]),
        "action_horizon": int(config["training"]["action_horizon"]),
        "query_stride": int(config["memory_adapter"]["state_token"]["query_stride"]),
        "execution_input_override": {"4": "item_1", "5": "item_2", "6": "item_3"},
        "forced_execution_memory_action_loss": "masked",
        "episodes": [
            {
                "episode_index": index,
                "source_episode": record.path.name,
                "annotation_path": str(record.annotation_path),
                "source_frames": record.total_frames,
                "source_fps": record.source_fps,
                "missing_observation_items": list(record.missing_observation_items),
                "intervals": [dataclasses.asdict(interval) for interval in record.intervals],
            }
            for index, record in enumerate(records)
        ],
    }
    (memory_dir / "phase_layout.json").write_text(
        json.dumps(phase_layout, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rmbench_dir = output_path / "meta" / "rmbench"
    _write_yaml(rmbench_dir / "key_state_config.yaml", config)
    command_text = " ".join(argv)
    (rmbench_dir / "convert_command.txt").write_text(command_text + "\n", encoding="utf-8")
    source_summary = {
        "dataset_root": str(_resolve_repo_path(config["dataset"]["source_dir"])),
        "required_tag": config["dataset"]["required_tag"],
        "tagged_episodes": len(records) + len(skipped),
        "converted_episodes": len(records),
        "skipped_episodes": len(skipped),
        "source_fps": config["dataset"]["source_fps"],
        "target_fps": config["dataset"]["target_fps"],
    }
    _write_yaml(rmbench_dir / "source_data_config.yaml", source_summary)
    (rmbench_dir / "source_data_command.txt").write_text(
        "selection: anno/tags.json contains drawer_sorting; labels: anno/sort.json\n", encoding="utf-8"
    )
    (rmbench_dir / "conversion_audit.json").write_text(
        json.dumps({"summary": source_summary, "skipped": skipped}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def convert(
    config: dict[str, Any],
    *,
    output_root: Path | None,
    overwrite: bool,
    debug_episodes: int | None,
    num_workers: int,
    video_codec: str,
    argv: list[str],
) -> Path:
    records, skipped = discover_episodes(config)
    if debug_episodes is not None:
        records = records[:debug_episodes]
    if not records:
        raise RuntimeError("No valid drawer_sorting episodes found")

    repo_id = str(config["dataset"]["repo_id"])
    output_path = (output_root or Path(HF_LEROBOT_HOME)) / repo_id
    if output_path.exists() or output_path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite")
        if output_path.is_symlink() or output_path.is_file():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)

    print(f"Tagged/valid episodes: {len(records)}; skipped: {len(skipped)}")
    print("SM2SM: 30 Hz source -> 15 Hz LeRobot; action_horizon metadata=30")
    dataset = _create_dataset(config, output_path)
    target_fps = int(config["dataset"]["target_fps"])
    query_stride = int(config["memory_adapter"]["state_token"]["query_stride"])
    target_size = (320, 240)

    transcode_tasks = [
        (episode_index, record, camera, filename)
        for episode_index, record in enumerate(records)
        for camera, filename in FILE_CAMERA_MAPPING.items()
    ]
    frame_counts: dict[int, dict[str, int]] = {}
    failures: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                transcode_video_ffmpeg,
                record.path / filename,
                output_path / "videos" / "chunk-000" / camera / f"episode_{episode_index:06d}.mp4",
                target_size,
                target_fps,
                video_codec,
            ): (episode_index, camera)
            for episode_index, record, camera, filename in transcode_tasks
        }
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Transcoding videos"):
            episode_index, camera = futures[future]
            try:
                frame_counts.setdefault(episode_index, {})[camera] = future.result()
            except Exception as exc:
                failures.setdefault(episode_index, []).append(f"{camera}: {exc}")

    if failures:
        details = "\n".join(f"episode {index}: {messages}" for index, messages in failures.items())
        raise RuntimeError(f"Video transcode failures; refusing a partial dataset:\n{details}")

    dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
    for episode_index, record in enumerate(tqdm.tqdm(records, desc="Building LeRobot rows")):
        target_frame_count = min(frame_counts[episode_index].values())
        state, action, input_ids, target_ids, target_valid, memory_action_valid = load_training_arrays(
            record, target_frame_count, target_fps, query_stride
        )
        dataset._video_frame_count = target_frame_count - 1  # noqa: SLF001
        for frame_index in range(target_frame_count - 1):
            dataset.add_frame(
                {
                    "face_view": dummy_image,
                    "left_wrist_view": dummy_image,
                    "right_wrist_view": dummy_image,
                    "state": state[frame_index],
                    "actions": np.concatenate([action[frame_index + 1, :28], action[frame_index, 28:]], axis=0),
                    "key_state_input_ids": input_ids[frame_index],
                    "key_state_target_ids": target_ids[frame_index],
                    "key_state_target_mask": target_valid[frame_index],
                    "memory_action_valid": memory_action_valid[frame_index],
                    "task": str(config["dataset"]["task_prompt"]),
                }
            )
        dataset.save_episode()

    dataset.finalize_video_info()
    images_dir = output_path / "images"
    if images_dir.is_dir():
        shutil.rmtree(images_dir)
    _write_metadata(output_path, config, records, skipped, argv)
    print(f"Wrote {len(records)} episodes to {output_path}")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-episodes", type=int)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--video-codec", choices=("h264", "av1"), default="h264")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    convert(
        config,
        output_root=args.output_root,
        overwrite=args.overwrite,
        debug_episodes=args.debug_episodes,
        num_workers=args.num_workers,
        video_codec=args.video_codec,
        argv=[Path(__file__).name, *(argv if argv is not None else sys.argv[1:])],
    )


if __name__ == "__main__":
    main()
