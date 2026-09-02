#!/usr/bin/env python3
"""Create OpenDM JSONL indices that reference RMBench HDF5 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

CAMERAS = (
    ("images_1", "/observations/images/cam_high"),
    ("images_2", "/observations/images/cam_left_wrist"),
    ("images_3", "/observations/images/cam_right_wrist"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("seen") or payload.get("unseen")
    if not prompts:
        raise ValueError(f"No seen/unseen prompt found in {path}")
    return str(prompts[0])


def episode_index(path: Path) -> int:
    return int(path.parent.name.removeprefix("episode_"))


def convert_episode(source_root: Path, source: Path, output: Path, prompt: str) -> int:
    relative_source = source.relative_to(source_root).as_posix()
    temporary = output.with_suffix(".jsonl.tmp")
    with (
        h5py.File(source, "r") as episode,
        temporary.open("w", encoding="utf-8") as stream,
    ):
        states = episode["/observations/qpos"]
        actions = episode["/action"]
        frame_count = int(states.shape[0])
        if actions.shape != states.shape or states.shape[1] != 14:
            raise ValueError(
                f"Unexpected state/action shapes in {source}: "
                f"state={states.shape}, action={actions.shape}"
            )
        for _, dataset_name in CAMERAS:
            if episode[dataset_name].shape[0] != frame_count:
                raise ValueError(f"Camera length mismatch in {source}: {dataset_name}")

        for frame_idx in range(frame_count):
            record = {
                key: {
                    "type": "hdf5",
                    "url": relative_source,
                    "dataset": dataset_name,
                    "frame_idx": frame_idx,
                }
                for key, dataset_name in CAMERAS
            }
            record.update(
                {
                    "state": states[frame_idx].astype(float).tolist(),
                    "action": actions[frame_idx].astype(float).tolist(),
                    "prompt": prompt,
                    "is_robot": True,
                }
            )
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(output)
    return frame_count


def main() -> None:
    args = parse_args()
    source_root = args.source_dir.resolve()
    files = sorted(source_root.glob("episode_*/episode_*.hdf5"), key=episode_index)
    if not files:
        raise FileNotFoundError(f"No episode HDF5 files found below {source_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt(args.prompt_file)
    index_data = {}
    total = 0
    for source in files:
        idx = episode_index(source)
        output = (args.output_dir / f"episode_{idx}.jsonl").resolve()
        count = convert_episode(source_root, source, output, prompt)
        index_data[str(output)] = count
        total += count
        print(f"episode {idx}: {count} frames -> {output}")
    (args.output_dir / "index_cache.json").write_text(
        json.dumps({"data": index_data}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"converted {len(files)} episodes / {total} frames")


if __name__ == "__main__":
    main()
