"""RMBench dataset registrations for DM0.5 fine-tuning."""

import os
from pathlib import Path

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

_RMBENCH_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_JSONL_DIR = _RMBENCH_ROOT / "policy/DM05/data/rmbench/swap_blocks"
_DEFAULT_HDF5_DIR = (
    _RMBENCH_ROOT / "policy/pi05/processed_data/swap_blocks-demo_clean-50"
)

register_dataset(
    {
        "swap_blocks": {
            "jsonl_dir": os.getenv("RMBENCH_DM05_JSONL_DIR", str(_DEFAULT_JSONL_DIR)),
            "image_dir": os.getenv(
                "RMBENCH_SWAP_BLOCKS_HDF5_DIR", str(_DEFAULT_HDF5_DIR)
            ),
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA_ROBOTWIN2,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2],
        },
    },
    prefix="rmbench",
)
