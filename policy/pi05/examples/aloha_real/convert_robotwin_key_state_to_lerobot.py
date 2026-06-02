"""Convert RMBench put_back_block data with key-state labels to LeRobot."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro


MAT_NAMES = ("left", "right", "front", "back")
CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


@dataclasses.dataclass(frozen=True)
class ConvertConfig:
    source_dir: Path
    repo_id: str
    episodes: int = 50
    instruction_type: Literal["seen", "unseen"] = "seen"
    mode: Literal["video", "image"] = "image"
    key_output_mode: Literal["per_step"] = "per_step"
    phase_input_policy: Literal["gt", "lag_after_boundary"] = "gt"
    mat_input_policy: Literal["unknown_until_wmat_end", "unknown_first_frame_only", "early_hash_mix"] = (
        "unknown_until_wmat_end"
    )
    wmat_margin_frames: int = 0
    mat_unknown_prob: float = 0.5
    phase_boundary_jitter_frames: int = 0
    phase_boundary_jitter_seed: int = 0
    lag_window_frames: int = 20
    dataset_config: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)


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


def _frame_hash_prob(episode_idx: int, frame_idx: int, seed: int) -> float:
    key = f"{seed}:{episode_idx}:{frame_idx}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _episode_key(episode_idx: int) -> str:
    return f"episode_{episode_idx}"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _segment_lengths(language_annotation: dict, episode_idx: int) -> list[int]:
    values = language_annotation[_episode_key(episode_idx)]
    lengths = [int(item[1]) for item in values]
    if len(lengths) != 10:
        raise ValueError(f"Expected 10 language segments for episode {episode_idx}, got {len(lengths)}")
    return lengths


def _phase_boundaries(lengths: list[int], episode_idx: int, cfg: ConvertConfig, action_frames: int) -> tuple[int, int]:
    b01 = sum(lengths[:3])
    b12 = sum(lengths[:7])

    jitter = int(cfg.phase_boundary_jitter_frames)
    if jitter > 0:
        rng = random.Random(cfg.phase_boundary_jitter_seed + episode_idx * 9973)
        b01 += rng.randint(-jitter, jitter)
        b12 += rng.randint(-jitter, jitter)

    b01 = int(np.clip(b01, 1, action_frames - 1))
    b12 = int(np.clip(b12, b01 + 1, action_frames))
    return b01, b12


def _phase_at(frame_idx: int, b01: int, b12: int) -> int:
    if frame_idx < b01:
        return 0
    if frame_idx < b12:
        return 1
    return 2


def _phase_input(frame_idx: int, b01: int, b12: int, cfg: ConvertConfig) -> int:
    phase = _phase_at(frame_idx, b01, b12)
    if cfg.phase_input_policy != "lag_after_boundary":
        return phase

    lag = max(0, int(cfg.lag_window_frames))
    if b01 <= frame_idx < b01 + lag:
        return 0
    if b12 <= frame_idx < b12 + lag:
        return 1
    return phase


def _mat_input(frame_idx: int, wmat_end: int, key_mat_id: int, episode_idx: int, cfg: ConvertConfig) -> int:
    if cfg.mat_input_policy == "unknown_first_frame_only":
        return 0 if frame_idx == 0 else key_mat_id

    if frame_idx >= wmat_end:
        return key_mat_id

    if cfg.mat_input_policy == "unknown_until_wmat_end":
        return 0

    if cfg.mat_input_policy == "early_hash_mix":
        p = _frame_hash_prob(episode_idx, frame_idx, cfg.phase_boundary_jitter_seed)
        return 0 if p < cfg.mat_unknown_prob else key_mat_id

    raise ValueError(f"Unknown mat_input_policy: {cfg.mat_input_policy}")


def _build_state_action(
    robot_state: np.ndarray,
    robot_action: np.ndarray,
    phase_input: int,
    mat_input: int,
    phase_target: int,
    mat_target: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros(32, dtype=np.float32)
    action = np.zeros(32, dtype=np.float32)
    state[:14] = robot_state.astype(np.float32)
    action[:14] = robot_action.astype(np.float32)
    state[14:17] = _one_hot(phase_input, 3)
    state[17:22] = _one_hot(mat_input, 5)
    action[14:17] = _one_hot(phase_target, 3)
    action[17:22] = _one_hot(mat_target, 5)
    return state, action


def _create_empty_dataset(cfg: ConvertConfig) -> LeRobotDataset:
    robot_dims = [
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
    key_dims = [
        "phase_move_to_center",
        "phase_press_button",
        "phase_move_back",
        "mat_unknown",
        "mat_left",
        "mat_right",
        "mat_front",
        "mat_back",
    ]
    names = robot_dims + key_dims + [f"pad_{i}" for i in range(10)]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (32,),
            "names": [names],
        },
        "action": {
            "dtype": "float32",
            "shape": (32,),
            "names": [names],
        },
    }
    for cam in CAMERA_MAP:
        features[f"observation.images.{cam}"] = {
            "dtype": cfg.mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    dataset_path = Path(HF_LEROBOT_HOME) / cfg.repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=50,
        robot_type="aloha",
        features=features,
        use_videos=cfg.dataset_config.use_videos,
        tolerance_s=cfg.dataset_config.tolerance_s,
        image_writer_processes=cfg.dataset_config.image_writer_processes,
        image_writer_threads=cfg.dataset_config.image_writer_threads,
        video_backend=cfg.dataset_config.video_backend,
    )


def _instruction_for_episode(cfg: ConvertConfig, episode_idx: int) -> str:
    path = cfg.source_dir / "instructions" / f"episode{episode_idx}.json"
    instructions = _load_json(path)[cfg.instruction_type]
    if not instructions:
        raise ValueError(f"No {cfg.instruction_type} instructions in {path}")
    return str(instructions[0])


def _populate_episode(
    dataset: LeRobotDataset,
    cfg: ConvertConfig,
    episode_idx: int,
    scene_info: dict,
    language_annotation: dict,
) -> dict:
    episode_path = cfg.source_dir / "data" / f"episode{episode_idx}.hdf5"
    info = scene_info[_episode_key(episode_idx)]["info"]
    key_mat_id = int(info["origin_mat_id"]) + 1
    if not 1 <= key_mat_id <= 4:
        raise ValueError(f"Invalid key_mat_id {key_mat_id} for episode {episode_idx}")

    lengths = _segment_lengths(language_annotation, episode_idx)
    instruction = _instruction_for_episode(cfg, episode_idx)

    with h5py.File(episode_path, "r") as ep:
        vector = ep["/joint_action/vector"][:].astype(np.float32)
        total_frames = int(vector.shape[0])
        action_frames = total_frames - 1
        b01, b12 = _phase_boundaries(lengths, episode_idx, cfg, action_frames)
        wmat_end = int(lengths[0] + cfg.wmat_margin_frames)
        wmat_end = int(np.clip(wmat_end, 1, action_frames))

        segment_total = sum(lengths)
        if segment_total not in (total_frames, action_frames):
            print(
                f"[WARN] episode={episode_idx} hdf5_frames={total_frames} "
                f"action_frames={action_frames} segment_total={segment_total}"
            )

        for frame_idx in range(action_frames):
            phase_in = _phase_input(frame_idx, b01, b12, cfg)
            phase_target = _phase_at(frame_idx + 1, b01, b12)
            mat_in = _mat_input(frame_idx, wmat_end, key_mat_id, episode_idx, cfg)
            state, action = _build_state_action(
                vector[frame_idx],
                vector[frame_idx + 1],
                phase_in,
                mat_in,
                phase_target,
                key_mat_id,
            )
            frame = {
                "observation.state": state,
                "action": action,
                "task": instruction,
            }
            for dst, src in CAMERA_MAP.items():
                frame[f"observation.images.{dst}"] = _decode_image(ep[f"/observation/{src}/rgb"][frame_idx])
            dataset.add_frame(frame)

    dataset.save_episode()
    return {
        "episode_idx": episode_idx,
        "frames": action_frames,
        "source_frames": total_frames,
        "segment_total": segment_total,
        "b01": b01,
        "b12": b12,
        "wmat_end": wmat_end,
        "mat_id": key_mat_id,
        "mat_name": MAT_NAMES[key_mat_id - 1],
    }


def convert(cfg: ConvertConfig) -> None:
    if cfg.key_output_mode != "per_step":
        raise ValueError("Only key_output_mode='per_step' is supported by this converter")

    scene_info = _load_json(cfg.source_dir / "scene_info.json")
    language_annotation = _load_json(cfg.source_dir / "language_annotation.json")
    dataset = _create_empty_dataset(cfg)

    summaries = []
    for episode_idx in tqdm.tqdm(range(cfg.episodes), desc=f"Converting {cfg.repo_id}"):
        summaries.append(_populate_episode(dataset, cfg, episode_idx, scene_info, language_annotation))

    dataset_path = Path(HF_LEROBOT_HOME) / cfg.repo_id
    meta_path = dataset_path / "meta" / "key_state_config.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "config": dataclasses.asdict(cfg),
                "episodes": summaries,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Wrote LeRobot dataset to {dataset_path}")
    print(f"Wrote key-state metadata to {meta_path}")


if __name__ == "__main__":
    convert(tyro.cli(ConvertConfig))
