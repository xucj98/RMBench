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


KEY_STATE_PHASE_SLICE = slice(14, 17)
KEY_STATE_MAT_SLICE = slice(17, 22)


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        self.key_state_enabled = "key_state_variant" in config.policy_metadata
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
    def _one_hot_from_logits(logits):
        value = np.zeros_like(logits, dtype=np.float32)
        value[int(np.argmax(logits))] = 1.0
        return value

    def update_key_state_from_action(self, action):
        if not self.key_state_enabled:
            return
        action = np.asarray(action, dtype=np.float32)
        self.key_state[:3] = self._one_hot_from_logits(action[KEY_STATE_PHASE_SLICE])
        self.key_state[3:8] = self._one_hot_from_logits(action[KEY_STATE_MAT_SLICE])

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
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            self._state_for_policy(state),
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
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
