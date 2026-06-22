"""Convert RMBench key-state demos to the zarr format used by DP."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import cv2
import h5py
import numpy as np
import yaml
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    return "clean" if not status else status


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


def _as_abs_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _one_hot(index: int, size: int) -> np.ndarray:
    value = np.zeros(size, dtype=np.float32)
    value[index] = 1.0
    return value


def _decode_image(encoded: bytes | np.bytes_) -> np.ndarray:
    data = np.frombuffer(encoded, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def _episode_key(episode_idx: int) -> str:
    return f"episode_{episode_idx}"


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


def _execution_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    execution = config.get("execution", [])
    if isinstance(execution, dict):
        return [execution]
    return list(execution or [])


def _schema_blocks(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    blocks = []
    if config.get("phase") is not None:
        blocks.append(("phase", config["phase"]))
    for execution_config in _execution_blocks(config):
        blocks.append((str(execution_config.get("name", "execution")), execution_config))
    for attr_config in config.get("attributes", []):
        blocks.append((str(attr_config.get("name", "attribute")), attr_config))
    return blocks


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
                f"one_hot dim size must match labels for {block_name}: "
                f"dim={[start, end]}, labels={labels}"
            )
        if _encoding(block) == "label_id" and width != 1:
            raise ValueError(
                f"label_id dim size must be 1 for {block_name}: "
                f"dim={[start, end]}"
            )
        if occupied[start:end].any():
            raise ValueError(f"Overlapping dim {block_name}: {[start, end]}")
        occupied[start:end] = 1


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
    return scene_info


def _default_zarr_path(config: dict[str, Any]) -> Path:
    dataset = config["dataset"]
    task = dataset["task"]
    episodes = int(dataset.get("episodes", 50))
    task_config = dataset.get("dp_task_config", "demo_clean_state_key_state")
    return PROJECT_ROOT / "policy" / "DP" / "data" / f"{task}-{task_config}-{episodes}.zarr"


def _zarr_path(config: dict[str, Any]) -> Path:
    zarr_path = config.get("dataset", {}).get("zarr_path")
    if zarr_path:
        return _as_abs_path(zarr_path)
    return _default_zarr_path(config)


def _episode_arrays(
    config: dict[str, Any],
    scene_info: dict[str, Any],
    episode_idx: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    source_dir = _as_abs_path(config["dataset"]["source_dir"])
    episode_path = source_dir / "data" / f"episode{episode_idx}.hdf5"
    info = scene_info[_episode_key(episode_idx)]["info"]

    head_camera_arrays = []
    state_arrays = []
    action_arrays = []

    with h5py.File(episode_path, "r") as ep:
        vector = ep["/joint_action/vector"][:].astype(np.float32)
        total_frames = int(vector.shape[0])
        action_frames = total_frames - 1
        robot_dim = int(config["state_layout"].get("robot_dim", 14))
        state_dim = int(config["state_layout"].get("state_dim", 32))

        per_step_blocks = []
        if config.get("phase") is not None:
            per_step_blocks.append(("phase", config["phase"]))
        per_step_blocks.extend((str(item.get("name", "execution")), item) for item in _execution_blocks(config))
        per_step_ranges = [
            (block_name, block_config, _range_labels(block_config, info, total_frames, block_name))
            for block_name, block_config in per_step_blocks
        ]

        for frame_idx in range(action_frames):
            state = np.zeros(state_dim, dtype=np.float32)
            action = np.zeros(state_dim, dtype=np.float32)
            state[:robot_dim] = vector[frame_idx, :robot_dim]
            action[:robot_dim] = vector[frame_idx + 1, :robot_dim]

            for _block_name, block_config, ranges in per_step_ranges:
                start, end = _dim(block_config)
                state[start:end] = _encoded_label(_label_at(frame_idx, ranges), block_config)
                action[start:end] = _encoded_label(_label_at(frame_idx + 1, ranges), block_config)

            for attr_config in config.get("attributes", []):
                start, end = _dim(attr_config)
                state_idx, target_idx = _attribute_indices(attr_config, info, total_frames, frame_idx)
                state[start:end] = _encoded_label(state_idx, attr_config)
                action[start:end] = _encoded_label(target_idx, attr_config)

            head_camera_arrays.append(_decode_image(ep["/observation/head_camera/rgb"][frame_idx]))
            state_arrays.append(state)
            action_arrays.append(action)

    summary = {
        "episode_idx": episode_idx,
        "frames": action_frames,
        "source_frames": total_frames,
    }
    for block_name, _block_config, ranges in per_step_ranges:
        summary[f"{block_name}_ranges"] = [
            {"label": label, "start_frame": start, "end_frame": end}
            for _idx, start, end, label in ranges
        ]
    return head_camera_arrays, state_arrays, action_arrays, summary


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


def convert(config: dict[str, Any], argv: list[str]) -> Path:
    config = copy.deepcopy(config)
    config.setdefault("dataset", {})
    config.setdefault("state_layout", {})
    config["dataset"].setdefault("episodes", 50)
    config["state_layout"].setdefault("state_dim", 32)
    config["state_layout"].setdefault("robot_dim", 14)

    _validate_layout(config)
    scene_info = _validate_source(config)

    save_dir = _zarr_path(config)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    all_head_camera = []
    all_state = []
    all_action = []
    episode_ends = []
    episode_summaries = []
    total_count = 0

    episodes = int(config["dataset"]["episodes"])
    for episode_idx in range(episodes):
        print(f"processing episode: {episode_idx + 1} / {episodes}", end="\r")
        head_camera, state, action, summary = _episode_arrays(config, scene_info, episode_idx)
        all_head_camera.extend(head_camera)
        all_state.extend(state)
        all_action.extend(action)
        total_count += len(action)
        episode_ends.append(total_count)
        episode_summaries.append(summary)
    print()

    head_camera_arrays = np.moveaxis(np.asarray(all_head_camera, dtype=np.uint8), -1, 1)
    state_arrays = np.asarray(all_state, dtype=np.float32)
    action_arrays = np.asarray(all_action, dtype=np.float32)
    episode_ends_arrays = np.asarray(episode_ends, dtype=np.int64)

    zarr_root = zarr.group(str(save_dir))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=(100, *head_camera_arrays.shape[1:]),
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=(100, state_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=(100, action_arrays.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    rmbench_meta_dir = save_dir / "meta" / "rmbench"
    rmbench_meta_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(rmbench_meta_dir / "key_state_config.yaml", config)
    _write_yaml(
        rmbench_meta_dir / "summary.yaml",
        {
            "zarr_path": str(save_dir.relative_to(PROJECT_ROOT)),
            "episodes": episodes,
            "frames": int(total_count),
            "state_dim": int(state_arrays.shape[1]),
            "action_dim": int(action_arrays.shape[1]),
            "image_shape": list(head_camera_arrays.shape[1:]),
            "episode_summaries": episode_summaries,
        },
    )
    _write_command(rmbench_meta_dir / "convert_command.txt", argv)
    _copy_source_metadata(config, rmbench_meta_dir)

    print(f"Wrote DP key-state zarr to {save_dir}")
    print(f"Wrote RMBench metadata to {rmbench_meta_dir}")
    return save_dir


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
    return config, [os.path.basename(__file__), *(argv if argv is not None else os.sys.argv[1:])]


if __name__ == "__main__":
    parsed_config, parsed_argv = parse_args()
    convert(parsed_config, parsed_argv)
