import numpy as np
import torch
import hydra
import dill
import sys, os
from pathlib import Path
import yaml

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(
        self,
        ckpt_file: str,
        n_obs_steps=None,
        n_action_steps=None,
        ddim_steps=None,
        key_state_config_path=None,
        key_state_update_mode="raw",
    ):
        if key_state_update_mode not in {"raw", "schema_latch"}:
            raise ValueError(f"Unsupported key_state_update_mode: {key_state_update_mode}")
        self.ckpt_file = str(ckpt_file)
        self.cfg = None
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        if n_obs_steps is None:
            n_obs_steps = int(self.cfg.n_obs_steps)
        if n_action_steps is None:
            n_action_steps = int(self.cfg.n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.runner = DPRunner(
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
        )
        if ddim_steps is not None:
            self.policy.set_inference_config(num_inference_steps=ddim_steps, use_ddim=True)
        self.robot_dim = 14
        self.state_dim = self._policy_state_dim()
        self.key_state_task = None
        self.key_state_schema = []
        self.key_state_update_mode = key_state_update_mode
        self.key_state_config_path = Path(key_state_config_path) if key_state_config_path else None
        self.key_state_enabled = bool(self.key_state_config_path and self.key_state_config_path.exists())
        if self.key_state_enabled:
            self._load_key_state_config(self.key_state_config_path)
        elif self._requires_key_state_metadata():
            raise FileNotFoundError(
                f"Missing key-state metadata for checkpoint {self.ckpt_file}: "
                f"{self.key_state_config_path}"
            )
        self.reset_key_state()

    def _policy_state_dim(self):
        try:
            return int(self.cfg.shape_meta.obs.agent_pos.shape[0])
        except Exception:
            return self.robot_dim

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset_obs(self):
        self.runner.reset_obs()
        self.reset_key_state()

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        self.cfg = cfg
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy

    def _requires_key_state_metadata(self):
        values = [self.ckpt_file]
        try:
            values.append(self.cfg.exp_name)
            values.append(self.cfg.task.dataset.zarr_path)
        except Exception:
            pass
        return any("key_state" in str(value) for value in values if value)

    def _load_key_state_config(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        config = payload.get("config", payload)
        layout = config.get("state_layout", {})
        self.state_dim = int(layout.get("state_dim", self.state_dim))
        self.robot_dim = int(layout.get("robot_dim", self.robot_dim))
        self.key_state_task = (config.get("dataset") or {}).get("task")

        schema = []
        phase = config.get("phase")
        if phase:
            schema.append(self._normalize_key_state_entry("phase", phase, "phase"))
        execution = config.get("execution", [])
        if isinstance(execution, dict):
            execution = [execution]
        for item in execution:
            schema.append(self._normalize_key_state_entry(item.get("name", "execution"), item, "execution"))
        for attr in config.get("attributes", []):
            schema.append(self._normalize_key_state_entry(attr.get("name", "attribute"), attr, "attribute"))

        if not schema:
            raise ValueError(f"No key-state entries found in key-state config: {path}")
        self.key_state_schema = schema

    def _normalize_key_state_entry(self, name, raw_entry, kind):
        dim = raw_entry.get("dim")
        if not isinstance(dim, list) or len(dim) != 2:
            raise ValueError(f"Invalid key-state dim for {name}: {dim}")
        start, end = int(dim[0]), int(dim[1])
        if start < self.robot_dim or end <= start or end > self.state_dim:
            raise ValueError(
                f"Invalid key-state dim for {name}: {dim}, "
                f"robot_dim={self.robot_dim}, state_dim={self.state_dim}"
            )
        labels = list(raw_entry.get("labels", []))
        if not labels:
            raise ValueError(f"Missing labels for key-state entry {name}")
        encoding = str(raw_entry.get("encoding", "one_hot"))
        if encoding not in {"one_hot", "label_id"}:
            raise ValueError(f"Unsupported key-state encoding for {name}: {encoding}")
        size = end - start
        if encoding == "one_hot" and len(labels) != size:
            raise ValueError(f"Label count mismatch for {name}: dim={dim}, labels={labels}")
        if encoding == "label_id" and size != 1:
            raise ValueError(f"label_id encoding requires a scalar dim for {name}: dim={dim}")
        return {
            "name": str(name),
            "kind": kind,
            "dim": (start, end),
            "labels": labels,
            "encoding": encoding,
        }

    @staticmethod
    def _one_hot_from_logits(logits):
        value = np.zeros_like(logits, dtype=np.float32)
        value[int(np.argmax(logits))] = 1.0
        return value

    @staticmethod
    def _clip_label_index(index, labels):
        return int(np.clip(index, 0, len(labels) - 1))

    def _initial_key_state_index(self, entry):
        labels = entry["labels"]
        if entry["kind"] == "attribute" and "unknown" in labels:
            return labels.index("unknown")
        return 0

    def _decode_key_state_index(self, entry, values):
        labels = entry["labels"]
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return self._initial_key_state_index(entry)
        if entry["encoding"] == "one_hot":
            return self._clip_label_index(int(np.argmax(values)), labels)
        if entry["encoding"] == "label_id":
            return self._clip_label_index(int(np.rint(float(values[0]))), labels)
        raise ValueError(f"Unsupported key-state encoding: {entry['encoding']}")

    def _write_key_state_value(self, entry, index):
        start, end = entry["dim"]
        index = self._clip_label_index(index, entry["labels"])
        self.key_state[start:end] = 0.0
        if entry["encoding"] == "one_hot":
            self.key_state[start + index] = 1.0
            return
        if entry["encoding"] == "label_id":
            self.key_state[start] = float(index)
            return
        raise ValueError(f"Unsupported key-state encoding: {entry['encoding']}")

    def reset_key_state(self):
        if not self.key_state_enabled:
            self.key_state = None
            return
        self.key_state = np.zeros(self.state_dim, dtype=np.float32)
        for entry in self.key_state_schema:
            self._write_key_state_value(entry, self._initial_key_state_index(entry))

    def state_for_policy(self, state):
        state = np.asarray(state, dtype=np.float32)
        if not self.key_state_enabled:
            return state
        policy_state = np.zeros(self.state_dim, dtype=np.float32)
        policy_state[:self.robot_dim] = state[:self.robot_dim]
        for entry in self.key_state_schema:
            start, end = entry["dim"]
            policy_state[start:end] = self.key_state[start:end]
        return policy_state

    def action_for_env(self, action):
        return np.asarray(action, dtype=np.float32)[:self.robot_dim]

    def update_key_state_from_action(self, action):
        if not self.key_state_enabled:
            return
        action = np.asarray(action, dtype=np.float32)
        if self.key_state_update_mode == "schema_latch":
            self._update_key_state_from_schema(action)
            return
        for entry in self.key_state_schema:
            start, end = entry["dim"]
            if action.shape[0] < end:
                raise ValueError(f"Action dim {action.shape[0]} is too small for key-state dim {start}:{end}")
            if entry["encoding"] == "one_hot":
                self.key_state[start:end] = self._one_hot_from_logits(action[start:end])
            else:
                self._write_key_state_value(entry, self._decode_key_state_index(entry, action[start:end]))

    def _update_key_state_from_schema(self, action):
        for entry in self.key_state_schema:
            start, end = entry["dim"]
            if action.shape[0] < end:
                raise ValueError(f"Action dim {action.shape[0]} is too small for key-state dim {start}:{end}")
            pred = self._decode_key_state_index(entry, action[start:end])
            current = self._decode_key_state_index(entry, self.key_state[start:end])
            if entry["kind"] == "phase":
                next_value = min(max(current, pred), current + 1)
            else:
                labels = entry["labels"]
                unknown_index = labels.index("unknown") if "unknown" in labels else None
                if unknown_index is not None:
                    next_value = pred if current == unknown_index and pred != unknown_index else current
                else:
                    next_value = pred
            self._write_key_state_value(entry, next_value)

    def get_eval_video_overlay(self):
        if not self.key_state_enabled:
            return None
        items = [
            {"label": "task", "value": self.key_state_task or Path(self.ckpt_file).parent.name},
            {"label": "update", "value": self.key_state_update_mode},
        ]
        for entry in self.key_state_schema:
            start, end = entry["dim"]
            index = self._decode_key_state_index(entry, self.key_state[start:end])
            labels = entry.get("labels", [])
            value = f"{labels[index]} [{index}]" if index < len(labels) else str(index)
            items.append({"label": entry["name"], "value": value})
        return {"title": "key-state", "items": items}
