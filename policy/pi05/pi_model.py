#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os


KEY_STATE_ACTION_OFFSET = 14
KEY_STATE_PHASE_SLICE = slice(KEY_STATE_ACTION_OFFSET, KEY_STATE_ACTION_OFFSET + 3)
KEY_STATE_MAT_SLICE = slice(KEY_STATE_ACTION_OFFSET + 3, KEY_STATE_ACTION_OFFSET + 8)
DEFAULT_KEY_STATE_SCHEMA = [
    {
        "name": "phase",
        "size": 3,
        "labels": ["move_to_center", "press_button", "move_back"],
        "update_rule": {"type": "monotonic", "max_step": 1},
    },
    {
        "name": "mat",
        "size": 5,
        "labels": ["unknown", "left", "right", "front", "back"],
        "update_rule": {"type": "latch_once_nonzero", "unknown_index": 0},
    },
]


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, key_state_update_mode="raw"):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        if key_state_update_mode not in {"raw", "schema_latch"}:
            raise ValueError(f"Unsupported key_state_update_mode: {key_state_update_mode}")
        self.key_state_update_mode = key_state_update_mode

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        self.policy_metadata = config.policy_metadata or {}
        self.key_state_enabled = "key_state_variant" in self.policy_metadata
        self.key_state_schema = self._get_key_state_schema(self.policy_metadata)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.reset_key_state()

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def reset_key_state(self):
        self.key_state = np.zeros(8, dtype=np.float32)
        self.key_state[0] = 1.0
        self.key_state[3] = 1.0

    @staticmethod
    def _get_key_state_schema(policy_metadata):
        schema = policy_metadata.get("key_state_schema")
        if not schema:
            return DEFAULT_KEY_STATE_SCHEMA
        normalized = []
        for item in schema:
            labels = item.get("labels", [])
            size = int(item.get("size", len(labels)))
            normalized.append({
                "name": item.get("name", "key_state"),
                "size": size,
                "labels": labels,
                "update_rule": item.get("update_rule", {"type": "raw"}),
            })
        return normalized

    @staticmethod
    def _one_hot_from_logits(logits):
        value = np.zeros_like(logits, dtype=np.float32)
        value[int(np.argmax(logits))] = 1.0
        return value

    def update_key_state_from_action(self, action):
        if not self.key_state_enabled:
            return
        action = np.asarray(action, dtype=np.float32)
        if self.key_state_update_mode == "schema_latch":
            self._update_key_state_from_schema(action)
            return
        self.key_state[:3] = self._one_hot_from_logits(action[KEY_STATE_PHASE_SLICE])
        self.key_state[3:8] = self._one_hot_from_logits(action[KEY_STATE_MAT_SLICE])

    def _update_key_state_from_schema(self, action):
        offset = 0
        for entry in self.key_state_schema:
            size = int(entry["size"])
            action_slice = action[KEY_STATE_ACTION_OFFSET + offset:KEY_STATE_ACTION_OFFSET + offset + size]
            state_slice = self.key_state[offset:offset + size]
            if action_slice.size == 0 or state_slice.size == 0:
                offset += size
                continue

            pred = int(np.argmax(action_slice))
            current = int(np.argmax(state_slice))
            rule = entry.get("update_rule", {"type": "raw"})
            rule_type = rule.get("type", "raw")

            if rule_type == "monotonic":
                max_step = int(rule.get("max_step", 1))
                next_value = min(max(current, pred), current + max_step)
            elif rule_type == "latch_once_nonzero":
                unknown_index = int(rule.get("unknown_index", 0))
                next_value = pred if current == unknown_index and pred != unknown_index else current
            elif rule_type == "raw":
                next_value = pred
            else:
                raise ValueError(f"Unsupported key-state update rule: {rule_type}")

            self.key_state[offset:offset + size] = 0.0
            self.key_state[offset + next_value] = 1.0
            offset += size

    def get_eval_video_overlay(self):
        if not self.key_state_enabled:
            return None
        items = [
            {"label": "variant", "value": self._display_variant_name()},
            {"label": "update", "value": self.key_state_update_mode},
        ]
        offset = 0
        for entry in self.key_state_schema:
            size = int(entry["size"])
            values = self.key_state[offset:offset + size]
            if values.size == 0:
                continue
            index = int(np.argmax(values))
            labels = entry.get("labels", [])
            if index < len(labels):
                value = f"{labels[index]} [{index}]"
            else:
                value = str(index)
            items.append({"label": entry["name"], "value": value})
            offset += size
        return {"title": "key-state", "items": items}

    def _display_variant_name(self):
        prefix = "pi0_put_back_block_key_state_"
        if self.model_name.startswith(prefix):
            return self.model_name[len(prefix):]
        return self.model_name

    def action_for_env(self, action):
        return np.asarray(action, dtype=np.float32)[:14]

    def _state_for_policy(self, state):
        state = np.asarray(state, dtype=np.float32)
        if not self.key_state_enabled:
            return state
        policy_state = np.zeros(32, dtype=np.float32)
        policy_state[:14] = state[:14]
        policy_state[14:22] = self.key_state
        return policy_state

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        policy_state = self._state_for_policy(state)
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": policy_state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.reset_key_state()
        print("successfully unset obs and language intruction")
