"""RMBench policy adapter for an OpenDM HTTP inference service."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image


@dataclass
class DM05Client:
    base_url: str
    action_horizon: int = 50
    timeout: float = 120.0
    robot_type: str = "Aloha RoboTwin2"

    def infer(self, instruction: str, observation: dict) -> np.ndarray:
        robot = observation["joint_action"]["vector"]
        cameras = observation["observation"]
        images = {
            "1": _encode_image(cameras["head_camera"]["rgb"]),
            "2": _encode_image(cameras["left_camera"]["rgb"]),
            "3": _encode_image(cameras["right_camera"]["rgb"]),
        }
        payload = {
            "observation": {
                "prompt": instruction,
                "robot_type": self.robot_type,
                "state": np.asarray(robot, dtype=np.float32).tolist(),
                "images": images,
            }
        }
        response = requests.post(self.base_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        actions = np.asarray(response.json()["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"DM0.5 returned invalid action shape {actions.shape}")
        return actions[: self.action_horizon]


def _encode_image(rgb: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        stream, format="JPEG", quality=95
    )
    return base64.b64encode(stream.getvalue()).decode("ascii")


def get_model(usr_args):
    return DM05Client(
        base_url=str(usr_args.get("base_url", "http://127.0.0.1:7891/v1/infer")),
        action_horizon=int(usr_args.get("action_horizon", 50)),
        timeout=float(usr_args.get("request_timeout", 120.0)),
        robot_type=str(usr_args.get("robot_type", "Aloha RoboTwin2")),
    )


def eval(TASK_ENV, model: DM05Client, observation: dict) -> None:
    actions = model.infer(TASK_ENV.get_instruction(), observation)
    for action in actions:
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        TASK_ENV.take_action(action, action_type="qpos")


def reset_model(model: DM05Client) -> None:
    return None
