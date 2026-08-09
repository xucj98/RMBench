"""Convert RMBench state-augmented demos to LeRobot datasets."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import yaml

CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
ROBOT_DIM_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SUPPORTED_ENCODINGS = {"one_hot", "label_id"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_yaml(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(value, f, allow_unicode=True, sort_keys=False)


def _parse_override_tokens(tokens: list[str]) -> dict[str, Any]:
    overrides = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if "=" in token:
            key, raw_value = token.split("=", 1)
            index += 1
        else:
            key = token.lstrip("-")
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing value for override: {token}")
            raw_value = tokens[index + 1]
            index += 2
        key = key.strip().lstrip("-")
        if not key:
            raise ValueError(f"Invalid override key in token: {token}")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def _apply_override(config: dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        _apply_override(config, key, value)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_status() -> str:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
    return status if status else "clean"


def _runtime_env() -> dict[str, str]:
    keys = ["CUDA_VISIBLE_DEVICES", "SAPIEN_RENDER_DEVICE", "PYTHONPATH"]
    return {key: os.environ[key] for key in keys if key in os.environ}


def _write_command(path: Path, argv: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"commit: {_git_commit()}\n")
        f.write(f"git_status: {_git_status()}\n")
        f.write(f"cwd: {Path.cwd()}\n")
        env = _runtime_env()
        if env:
            f.write("env:\n")
            for key, value in env.items():
                f.write(f"  {key}={value}\n")
        f.write("command:\n")
        f.write(f"  {' '.join(shlex.quote(item) for item in argv)}\n")


def _one_hot(index: int, size: int) -> np.ndarray:
    value = np.zeros(size, dtype=np.float32)
    value[index] = 1.0
    return value


def _decode_image(encoded: bytes | np.bytes_) -> np.ndarray:
    data = np.frombuffer(encoded, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return cv2.resize(image, (640, 480))


def _episode_key(episode_idx: int) -> str:
    return f"episode_{episode_idx}"


def _as_abs_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _get_nested(value: Any, path: list[str]) -> Any:
    cursor = value
    for part in path:
        if isinstance(cursor, dict):
            if part not in cursor:
                raise KeyError(".".join(path))
            cursor = cursor[part]
        elif isinstance(cursor, list):
            try:
                cursor = cursor[int(part)]
            except (ValueError, IndexError) as exc:
                raise KeyError(".".join(path)) from exc
        else:
            raise KeyError(".".join(path))
    return cursor


def _stage_map(info: dict[str, Any]) -> dict[str, dict[str, int]]:
    stages = {}
    for item in info.get("micro_stages", []):
        name = str(item["name"])
        stages[name] = {
            "start_frame": int(item["start_frame"]),
            "end_frame": int(item["end_frame"]),
        }
    return stages


def _resolve_ref(value: Any, info: dict[str, Any], total_frames: int) -> Any:
    if not isinstance(value, str):
        return value
    if value == "episode_start":
        return 0
    if value == "episode_end":
        return total_frames
    if value.startswith("task_facts."):
        return _get_nested(info["task_facts"], value.split(".")[1:])
    if value.startswith("micro_stages."):
        parts = value.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid micro stage reference: {value}")
        stage_name, field = parts[1], parts[2]
        stages = _stage_map(info)
        if stage_name not in stages:
            raise KeyError(value)
        if field not in stages[stage_name]:
            raise KeyError(value)
        return stages[stage_name][field]
    return value


def _resolve_window(window: list[Any], info: dict[str, Any], total_frames: int) -> tuple[int, int]:
    if len(window) != 2:
        raise ValueError(f"Window must contain [start, end], got: {window}")
    start = int(_resolve_ref(window[0], info, total_frames))
    end = int(_resolve_ref(window[1], info, total_frames))
    if not (0 <= start < end <= total_frames):
        raise ValueError(f"Invalid window [{start}, {end}) for total_frames={total_frames}")
    return start, end


def _labels(config: dict[str, Any]) -> list[str]:
    labels = config.get("labels")
    if not labels:
        raise ValueError(f"Missing labels in config block: {config}")
    return [str(item) for item in labels]


def _encoding(config: dict[str, Any]) -> str:
    encoding = str(config.get("encoding", "one_hot"))
    if encoding not in SUPPORTED_ENCODINGS:
        raise ValueError(f"Unsupported encoding {encoding!r}; expected one of {sorted(SUPPORTED_ENCODINGS)}")
    return encoding


def _dim(config: dict[str, Any]) -> tuple[int, int]:
    dim = config.get("dim")
    if not isinstance(dim, list) or len(dim) != 2:
        raise ValueError(f"dim must be [start, end], got: {dim}")
    return int(dim[0]), int(dim[1])


def _label_index(labels: list[str], value: Any) -> int:
    value = str(value)
    if value not in labels:
        raise ValueError(f"Value {value!r} not found in labels {labels}")
    return labels.index(value)


def _encoded_label(index: int, block_config: dict[str, Any]) -> np.ndarray:
    start, end = _dim(block_config)
    width = end - start
    encoding = _encoding(block_config)
    if encoding == "one_hot":
        return _one_hot(index, width)
    if encoding == "label_id":
        return np.array([float(index)], dtype=np.float32)
    raise ValueError(f"Unsupported encoding: {encoding}")


def _range_labels(
    block_config: dict[str, Any],
    info: dict[str, Any],
    total_frames: int,
    block_name: str,
) -> list[tuple[int, int, int, str]]:
    labels = _labels(block_config)
    ranges = []
    for item in block_config.get("ranges", []):
        try:
            label = str(_resolve_ref(item["label"], info, total_frames))
            start, end = _resolve_window(item["window"], info, total_frames)
        except KeyError:
            if item.get("optional", False):
                continue
            raise
        if label not in labels:
            raise ValueError(f"{block_name} range label {label!r} not in labels {labels}")
        ranges.append((_label_index(labels, label), start, end, label))
    ranges.sort(key=lambda item: item[1])
    if not ranges:
        raise ValueError(f"No {block_name} ranges resolved")
    return ranges


def _label_at(frame_idx: int, ranges: list[tuple[int, int, int, str]]) -> int:
    for index, start, end, _label in ranges:
        if start <= frame_idx < end:
            return index
    if frame_idx < ranges[0][1]:
        return ranges[0][0]
    return ranges[-1][0]


def _attribute_indices(
    attr_config: dict[str, Any],
    info: dict[str, Any],
    total_frames: int,
    frame_idx: int,
) -> tuple[int, int]:
    labels = _labels(attr_config)
    transitions = attr_config.get("transitions", [])
    if not transitions:
        raise ValueError(f"Attribute {attr_config.get('name')} has no transitions")

    state_value = _resolve_ref(transitions[0]["from_value"], info, total_frames)
    target_value = state_value
    for transition in transitions:
        from_value = _resolve_ref(transition["from_value"], info, total_frames)
        to_value = _resolve_ref(transition["to_value"], info, total_frames)
        start, end = _resolve_window(transition["update_window"], info, total_frames)
        if frame_idx < start:
            break
        if start <= frame_idx < end:
            state_value = from_value
            target_value = to_value
            break
        state_value = to_value
        target_value = to_value

    return _label_index(labels, state_value), _label_index(labels, target_value)


def _structured_state_token_resolvers(
    structured_config: dict[str, Any], info: dict[str, Any], total_frames: int
) -> list[tuple[str, dict[str, Any], list[tuple[int, int, int, str]] | None]] | None:
    fields = list(structured_config.get("fields", []))
    if not fields:
        raise ValueError("structured_state_tokens.fields must not be empty")
    declarative = all(field.get("ranges") or field.get("transitions") for field in fields)
    if not declarative:
        field_names = [str(field.get("name", "state")) for field in fields]
        legacy_names = ["phase", "empty_mat_side", "button_press_status"]
        if field_names != legacy_names:
            raise ValueError(
                f"non-declarative structured state-token fields are supported only for {legacy_names}, got {field_names}"
            )
        # Backward-compatible path for the original rearrange-blocks config,
        # whose three labels are resolved by the button-aware code below.
        return None
    resolvers = []
    for field in fields:
        name = str(field.get("name", "state"))
        has_ranges = bool(field.get("ranges"))
        has_transitions = bool(field.get("transitions"))
        if has_ranges == has_transitions:
            raise ValueError(f"structured state-token field {name!r} needs exactly one of ranges or transitions")
        ranges = _range_labels(field, info, total_frames, name) if has_ranges else None
        resolvers.append((name, field, ranges))
    return resolvers


def _structured_state_token_ids(
    resolvers: list[tuple[str, dict[str, Any], list[tuple[int, int, int, str]] | None]],
    info: dict[str, Any],
    total_frames: int,
    frame_idx: int,
) -> np.ndarray:
    ids = []
    for _name, field, ranges in resolvers:
        if ranges is not None:
            ids.append(_label_at(frame_idx, ranges))
        else:
            # The target side of an acquisition transition is the semantic
            # state predicted from the current observation. The carried input
            # comes from this target at the previous query time.
            _state_index, target_index = _attribute_indices(field, info, total_frames, frame_idx)
            ids.append(target_index)
    return np.asarray(ids, dtype=np.int64)


def _execution_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    execution = config.get("execution", [])
    if isinstance(execution, dict):
        return [execution]
    return list(execution or [])


def _schema_blocks(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    blocks = []
    if config.get("phase") is not None:
        blocks.append(("phase", config["phase"]))
    blocks.extend(
        (str(execution_config.get("name", "execution")), execution_config)
        for execution_config in _execution_blocks(config)
    )
    blocks.extend(
        (str(attr_config.get("name", "attribute")), attr_config) for attr_config in config.get("attributes", [])
    )
    return blocks


def _set_feature_names(names: list[str], block_name: str, block_config: dict[str, Any]) -> None:
    start, end = _dim(block_config)
    labels = _labels(block_config)
    encoding = _encoding(block_config)
    if encoding == "one_hot":
        if end - start != len(labels):
            raise ValueError(f"{block_name} dim size does not match labels")
        for offset, label in enumerate(labels):
            names[start + offset] = f"{block_name}_{label}"
    else:
        if end - start != 1:
            raise ValueError(f"label_id {block_name} encoding must use exactly one dim")
        names[start] = f"{block_name}_label_id"


def _feature_names(config: dict[str, Any]) -> list[str]:
    state_dim = int(config["state_layout"].get("state_dim", 32))
    robot_dim = int(config["state_layout"].get("robot_dim", 14))
    names = [f"dim_{idx}" for idx in range(state_dim)]
    for idx in range(min(robot_dim, len(ROBOT_DIM_NAMES))):
        names[idx] = ROBOT_DIM_NAMES[idx]

    for block_name, block_config in _schema_blocks(config):
        _set_feature_names(names, block_name, block_config)

    for idx, name in enumerate(names):
        if name == f"dim_{idx}":
            names[idx] = f"pad_{idx}"
    return names


def _validate_layout(config: dict[str, Any]) -> None:
    state_dim = int(config["state_layout"].get("state_dim", 32))
    occupied = np.zeros(state_dim, dtype=np.int32)
    robot_dim = int(config["state_layout"].get("robot_dim", 14))
    occupied[:robot_dim] = 1
    for block_name, block in _schema_blocks(config):
        start, end = _dim(block)
        labels = _labels(block)
        width = end - start
        if start < 0 or end > state_dim or start >= end:
            raise ValueError(f"Invalid dim {block_name}: {[start, end]}")
        if _encoding(block) == "one_hot" and width != len(labels):
            raise ValueError(
                f"one_hot dim size must match labels for {block_name}: " f"dim={[start, end]}, labels={labels}"
            )
        if _encoding(block) == "label_id" and width != 1:
            raise ValueError(f"label_id dim size must be 1 for {block_name}: " f"dim={[start, end]}")
        if occupied[start:end].any():
            raise ValueError(f"Overlapping dim {block_name}: {[start, end]}")
        occupied[start:end] = 1


def _create_empty_dataset(config: dict[str, Any]) -> LeRobotDataset:
    _validate_layout(config)
    dataset_config = config.get("dataset_config", {})
    dataset = config["dataset"]
    state_dim = int(config["state_layout"].get("state_dim", 32))
    mode = dataset.get("mode", "image")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [_feature_names(config)],
        },
        "action": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [_feature_names(config)],
        },
    }
    structured_config = config.get("structured_state_tokens")
    if structured_config is not None:
        fields = list(structured_config.get("fields", []))
        if not fields:
            raise ValueError("structured_state_tokens.fields must not be empty")
        field_names = [str(field.get("name", "state")) for field in fields]
        if len(set(field_names)) != len(field_names):
            raise ValueError(f"structured state-token field names must be unique: {field_names}")
        for field in fields:
            _labels(field)
        field_count = len(fields)
        features.update(
            {
                "observation.key_state_input_ids": {
                    "dtype": "int64",
                    "shape": (field_count,),
                    "names": field_names,
                },
                "observation.key_state_target_ids": {
                    "dtype": "int64",
                    "shape": (field_count,),
                    "names": field_names,
                },
                "observation.key_state_target_mask": {
                    "dtype": "bool",
                    "shape": (field_count,),
                    "names": field_names,
                },
            }
        )
        if "button_press_down_segment" in structured_config:
            features["observation.key_state_guard_offset"] = {
                "dtype": "int64",
                "shape": (1,),
                "names": ["button_press_confirmed_minus_frame"],
            }
    for cam in CAMERA_MAP:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset_path = Path(HF_LEROBOT_HOME) / dataset["repo_id"]
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=dataset["repo_id"],
        fps=int(dataset.get("fps", 50)),
        robot_type=dataset.get("robot_type", "aloha"),
        features=features,
        use_videos=bool(dataset_config.get("use_videos", True)),
        tolerance_s=float(dataset_config.get("tolerance_s", 0.0001)),
        image_writer_processes=int(dataset_config.get("image_writer_processes", 10)),
        image_writer_threads=int(dataset_config.get("image_writer_threads", 5)),
        video_backend=dataset_config.get("video_backend"),
    )


def _instruction_for_episode(config: dict[str, Any], episode_idx: int) -> str:
    source_dir = _as_abs_path(config["dataset"]["source_dir"])
    instruction_type = config["dataset"].get("instruction_type", "seen")
    path = source_dir / "instructions" / f"episode{episode_idx}.json"
    instructions = _load_json(path)[instruction_type]
    if not instructions:
        raise ValueError(f"No {instruction_type} instructions in {path}")
    return str(instructions[0])


def _populate_episode(
    dataset: LeRobotDataset,
    config: dict[str, Any],
    episode_idx: int,
    scene_info: dict[str, Any],
    language_annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_dir = _as_abs_path(config["dataset"]["source_dir"])
    episode_path = source_dir / "data" / f"episode{episode_idx}.hdf5"
    info = scene_info[_episode_key(episode_idx)]["info"]
    instruction = _instruction_for_episode(config, episode_idx)

    with h5py.File(episode_path, "r") as ep:
        vector = ep["/joint_action/vector"][:].astype(np.float32)
        total_frames = int(vector.shape[0])
        action_frames = total_frames - 1
        structured_config = config.get("structured_state_tokens")
        structured_resolvers = None
        guard_frame = None
        if structured_config is not None:
            structured_resolvers = _structured_state_token_resolvers(structured_config, info, total_frames)
            if "button_press_down_segment" in structured_config:
                if language_annotation is None:
                    raise ValueError("button-aware structured_state_tokens requires language_annotation.json")
                segments = language_annotation[_episode_key(episode_idx)]
                guard_segment = int(structured_config["button_press_down_segment"])
                if len(segments) != 11 or not (0 <= guard_segment < len(segments)):
                    raise ValueError(f"Unexpected language segments for episode {episode_idx}: {len(segments)}")
                annotated_frames = sum(int(segment[1]) for segment in segments)
                # Language annotations count observation frames, while the converter
                # emits N-1 transitions. Historical datasets contain both counting
                # conventions, so accept either exact total.
                if annotated_frames not in {action_frames, total_frames}:
                    raise ValueError(
                        f"Language segment durations ({annotated_frames}) match neither observation frames "
                        f"({total_frames}) nor action frames ({action_frames}) for episode {episode_idx}"
                    )
                guard_frame = sum(int(segment[1]) for segment in segments[: guard_segment + 1])
        per_step_blocks = []
        if config.get("phase") is not None:
            per_step_blocks.append(("phase", config["phase"]))
        per_step_blocks.extend((str(item.get("name", "execution")), item) for item in _execution_blocks(config))
        per_step_ranges = [
            (block_name, block_config, _range_labels(block_config, info, total_frames, block_name))
            for block_name, block_config in per_step_blocks
        ]

        for frame_idx in range(action_frames):
            state = np.zeros(int(config["state_layout"].get("state_dim", 32)), dtype=np.float32)
            action = np.zeros_like(state)
            robot_dim = int(config["state_layout"].get("robot_dim", 14))
            state[:robot_dim] = vector[frame_idx, :robot_dim].astype(np.float32)
            action[:robot_dim] = vector[frame_idx + 1, :robot_dim].astype(np.float32)

            for _block_name, block_config, ranges in per_step_ranges:
                start, end = _dim(block_config)
                state[start:end] = _encoded_label(_label_at(frame_idx, ranges), block_config)
                action[start:end] = _encoded_label(_label_at(frame_idx + 1, ranges), block_config)

            for attr_config in config.get("attributes", []):
                attr_start, attr_end = _dim(attr_config)
                attr_state_idx, attr_target_idx = _attribute_indices(attr_config, info, total_frames, frame_idx)
                state[attr_start:attr_end] = _encoded_label(attr_state_idx, attr_config)
                action[attr_start:attr_end] = _encoded_label(attr_target_idx, attr_config)

            frame = {
                "observation.state": state,
                "action": action,
                "task": instruction,
            }
            if structured_config is not None:
                stride = int(structured_config.get("query_stride", 20))
                field_count = len(structured_config["fields"])

                def state_token_ids(index: int) -> np.ndarray:
                    if structured_resolvers is not None:
                        return _structured_state_token_ids(structured_resolvers, info, total_frames, index)
                    # Legacy rearrange-blocks button-aware labels.
                    block1_end = int(_resolve_ref("micro_stages.block1_place.end_frame", info, total_frames))
                    press_return_end = int(_resolve_ref("micro_stages.press_return.end_frame", info, total_frames))
                    phase = 0 if index < block1_end else (1 if index < press_return_end else 2)
                    side_name = str(_resolve_ref("task_facts.empty_mat_side", info, total_frames))
                    side = {"left": 1, "right": 2}[side_name]
                    button = 0 if phase != 1 else 1 if index < guard_frame else 2
                    return np.asarray([phase, side, button], dtype=np.int64)

                previous_ids = (
                    state_token_ids(frame_idx - stride)
                    if frame_idx - stride >= 0
                    else np.zeros(field_count, dtype=np.int64)
                )
                frame.update(
                    {
                        "observation.key_state_input_ids": previous_ids,
                        "observation.key_state_target_ids": state_token_ids(frame_idx),
                        "observation.key_state_target_mask": np.ones(field_count, dtype=np.bool_),
                    }
                )
                if guard_frame is not None:
                    frame["observation.key_state_guard_offset"] = np.asarray([guard_frame - frame_idx], dtype=np.int64)
            for dst, src in CAMERA_MAP.items():
                frame[f"observation.images.{dst}"] = _decode_image(ep[f"/observation/{src}/rgb"][frame_idx])
            dataset.add_frame(frame)

    dataset.save_episode()
    summary = {
        "episode_idx": episode_idx,
        "frames": action_frames,
        "source_frames": total_frames,
    }
    for block_name, _block_config, ranges in per_step_ranges:
        summary[f"{block_name}_ranges"] = [
            {"label": label, "start_frame": start, "end_frame": end} for _idx, start, end, label in ranges
        ]
    if guard_frame is not None:
        summary["button_press_confirmed"] = {
            "frame": guard_frame,
            "source": "language_annotation.segment_4.end",
        }
    return summary


def _copy_source_metadata(config: dict[str, Any], rmbench_meta_dir: Path) -> None:
    source_dir = _as_abs_path(config["dataset"]["source_dir"])
    source_meta = source_dir / "metadata"
    required = {
        "config.yaml": "source_data_config.yaml",
        "command.txt": "source_data_command.txt",
    }
    for source_name, target_name in required.items():
        source_path = source_meta / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source metadata file: {source_path}")
        shutil.copy2(source_path, rmbench_meta_dir / target_name)


def _validate_source(config: dict[str, Any]) -> dict[str, Any]:
    source_dir = _as_abs_path(config["dataset"]["source_dir"])
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    scene_info = _load_json(source_dir / "scene_info.json")
    requested_episodes = int(config["dataset"].get("episodes", 50))
    data_files = sorted((source_dir / "data").glob("episode*.hdf5"))
    if len(data_files) < requested_episodes:
        raise ValueError(
            f"Source has {len(data_files)} hdf5 episodes, but config requests {requested_episodes}: {source_dir}"
        )
    for episode_idx in range(requested_episodes):
        instruction_path = source_dir / "instructions" / f"episode{episode_idx}.json"
        if not instruction_path.exists():
            raise FileNotFoundError(f"Missing instruction file: {instruction_path}")
    return scene_info


def convert(config: dict[str, Any], argv: list[str]) -> None:
    config = copy.deepcopy(config)
    config.setdefault("dataset", {})
    config["dataset"].setdefault("instruction_type", "seen")
    config["dataset"].setdefault("mode", "image")
    config["dataset"].setdefault("fps", 50)
    config["dataset"].setdefault("robot_type", "aloha")
    config.setdefault("dataset_config", {})

    scene_info = _validate_source(config)
    source_dir = _as_abs_path(config["dataset"]["source_dir"])
    language_annotation_path = source_dir / "language_annotation.json"
    language_annotation = _load_json(language_annotation_path) if language_annotation_path.exists() else None
    dataset = _create_empty_dataset(config)

    for episode_idx in tqdm.tqdm(
        range(int(config["dataset"]["episodes"])), desc=f"Converting {config['dataset']['repo_id']}"
    ):
        _populate_episode(dataset, config, episode_idx, scene_info, language_annotation)

    dataset_path = Path(HF_LEROBOT_HOME) / config["dataset"]["repo_id"]
    rmbench_meta_dir = dataset_path / "meta" / "rmbench"
    rmbench_meta_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(rmbench_meta_dir / "key_state_config.yaml", config)
    _write_command(rmbench_meta_dir / "convert_command.txt", argv)
    _copy_source_metadata(config, rmbench_meta_dir)

    print(f"Wrote LeRobot dataset to {dataset_path}")
    print(f"Wrote RMBench metadata to {rmbench_meta_dir}")


def parse_args(argv: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args(argv)

    config = _load_yaml(Path(args.config))
    overrides = _parse_override_tokens(args.overrides)
    _apply_overrides(config, overrides)
    config["_runtime"] = {
        "config_path": args.config,
        "overrides": overrides,
    }
    return config, [Path(__file__).name, *(argv if argv is not None else sys.argv[1:])]


if __name__ == "__main__":
    parsed_config, parsed_argv = parse_args()
    convert(parsed_config, parsed_argv)
