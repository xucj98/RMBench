#!/usr/bin/env python3
"""Extract demo videos from HDF5 data for each task."""
import os
import h5py
import cv2
import numpy as np
from pathlib import Path

TASKS = [
    "observe_and_pickup",
    "put_back_block",
    "rearrange_blocks",
    "swap_blocks",
    "swap_T",
    "battery_try",
    "blocks_ranking_try",
    "cover_blocks",
    "press_button",
]

BASE_DIR = Path("/root/projects/RMBench/data/data")
OUTPUT_DIR = Path("/root/projects/RMBench/task_videos")
OUTPUT_DIR.mkdir(exist_ok=True)

FPS = 30

for task in TASKS:
    hdf5_dir = BASE_DIR / task / "demo_clean" / "data"
    if not hdf5_dir.exists():
        print(f"Skipping {task}: no hdf5 dir")
        continue

    hdf5_files = sorted(hdf5_dir.glob("episode*.hdf5"))
    if not hdf5_files:
        print(f"Skipping {task}: no hdf5 files")
        continue

    # Use first episode for demo video
    hdf5_path = hdf5_files[0]
    output_path = OUTPUT_DIR / f"{task}_demo.mp4"

    print(f"Processing {task} from {hdf5_path}...")

    with h5py.File(hdf5_path, "r") as f:
        head_cam = f["/observation/head_camera/rgb"]
        n_frames = len(head_cam)

        if n_frames == 0:
            print(f"  {task}: no frames")
            continue

        # Decode first frame to get size
        first_frame = cv2.imdecode(np.frombuffer(head_cam[0], np.uint8), cv2.IMREAD_COLOR)
        h, w = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (w, h))

        for i in range(n_frames):
            frame = cv2.imdecode(np.frombuffer(head_cam[i], np.uint8), cv2.IMREAD_COLOR)
            writer.write(frame)

        writer.release()
        print(f"  Saved {output_path} ({n_frames} frames)")

print("\nAll videos extracted!")
