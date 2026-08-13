#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from collections import deque
import dataclasses
import os
from pathlib import Path

import numpy as np
import yaml

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CHECKPOINT_ROOT = Path("policy/pi05/checkpoints")
KEY_STATE_CONFIG_RELATIVE_PATH = Path("metadata/rmbench_data_meta/key_state_config.yaml")


class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        key_state_update_mode="raw",
        state_token_rollout_mode="predicted",
        key_state_rollout_mode="predicted",
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        if key_state_update_mode not in {"raw", "schema_latch"}:
            raise ValueError(f"Unsupported key_state_update_mode: {key_state_update_mode}")
        self.key_state_update_mode = key_state_update_mode
        if state_token_rollout_mode not in {"predicted", "oracle"}:
            raise ValueError(f"Unsupported state_token_rollout_mode: {state_token_rollout_mode}")
        self.state_token_rollout_mode = state_token_rollout_mode
        if key_state_rollout_mode not in {"predicted", "oracle"}:
            raise ValueError(f"Unsupported key_state_rollout_mode: {key_state_rollout_mode}")
        self.key_state_rollout_mode = key_state_rollout_mode

        self.checkpoint_dir = CHECKPOINT_ROOT / self.train_config_name / self.model_name / str(self.checkpoint_id)
        self.checkpoint_run_dir = self.checkpoint_dir.parent
        specified_path = self.checkpoint_dir / "assets"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = self._load_checkpoint_train_config(_config.get_config(self.train_config_name))
        self.policy_metadata = config.policy_metadata or {}
        self.state_token_enabled = getattr(config.model, "key_state_token_mode", "disabled") != "disabled"
        self.state_token_mode = getattr(config.model, "key_state_token_mode", "disabled")
        self.state_token_schema = []
        self.state_token_ids = None
        self.state_token_prediction_ids = None
        self.state_history_size = int(getattr(config.data, "state_history_size", 0))
        self.state_sequence_length = self.state_history_size + 1
        self.state_history = deque(maxlen=self.state_sequence_length)
        self.robot_dim = 14
        self.state_dim = 14
        self.key_state_task = None
        self.key_state_schema = []
        self.key_state_config_path = self.checkpoint_run_dir / KEY_STATE_CONFIG_RELATIVE_PATH
        self.key_state_enabled = self.key_state_config_path.exists() and not self.state_token_enabled
        if self.state_token_enabled:
            if not self.key_state_config_path.exists():
                raise FileNotFoundError(
                    f"Missing state-token metadata for checkpoint {self.checkpoint_run_dir}: "
                    f"{self.key_state_config_path}"
                )
            self._load_state_token_config(
                self.key_state_config_path,
                config.model.key_state_num_values,
                field_indices=self.policy_metadata.get("key_state_field_indices"),
            )
        elif self.key_state_enabled:
            self._load_key_state_config(self.key_state_config_path)
        elif self._requires_key_state_metadata():
            raise FileNotFoundError(
                f"Missing key-state metadata for checkpoint {self.checkpoint_run_dir}: "
                f"{self.key_state_config_path}"
            )
        if self.state_token_rollout_mode == "oracle" and self.state_token_mode != "serial":
            raise ValueError("oracle state-token rollout requires a serial state-token checkpoint")
        if self.key_state_rollout_mode == "oracle" and not self.key_state_enabled:
            raise ValueError("oracle dense key-state rollout requires a full key-state checkpoint")

        self.policy = _policy_config.create_trained_policy(
            config,
            str(self.checkpoint_dir),
            robotwin_repo_id=assets_id,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step
        self.reset_key_state()

    def _load_checkpoint_train_config(self, config):
        metadata_path = self.checkpoint_run_dir / "metadata" / "train_config.yaml"
        if not metadata_path.exists():
            return config

        with metadata_path.open("r", encoding="utf-8") as f:
            saved_config = yaml.safe_load(f) or {}
        saved_name = saved_config.get("name")
        if saved_name and saved_name != self.train_config_name:
            raise ValueError(
                f"Checkpoint train config mismatch for {metadata_path}: "
                f"metadata={saved_name!r}, requested={self.train_config_name!r}"
            )

        saved_model = saved_config.get("model") or {}
        saved_mode = saved_model.get("key_state_token_mode")
        if saved_mode is None:
            return config
        saved_num_values = tuple(
            int(value) for value in saved_model.get("key_state_num_values", config.model.key_state_num_values)
        )
        model_config = dataclasses.replace(
            config.model,
            key_state_token_mode=str(saved_mode),
            key_state_num_values=saved_num_values,
        )
        policy_metadata = saved_config.get("policy_metadata", config.policy_metadata)
        return dataclasses.replace(config, model=model_config, policy_metadata=policy_metadata)

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    def reset_key_state(self):
        if self.state_token_enabled:
            self.state_token_ids = np.zeros(len(self.state_token_schema), dtype=np.int32)
            self.state_token_prediction_ids = self.state_token_ids.copy()
        if not self.key_state_enabled:
            self.key_state = None
            return

        self.key_state = np.zeros(self.state_dim, dtype=np.float32)
        for entry in self.key_state_schema:
            self._write_key_state_value(entry, self._initial_key_state_index(entry))

    def _load_state_token_config(self, path, expected_num_values, *, field_indices=None):
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        config = payload.get("config", payload)
        fields = (config.get("structured_state_tokens") or {}).get("fields", [])
        if not fields:
            raise ValueError(f"No structured_state_tokens.fields found in state-token config: {path}")
        if field_indices is not None:
            indices = tuple(int(index) for index in field_indices)
            if not indices or min(indices) < 0 or max(indices) >= len(fields):
                raise ValueError(f"Invalid state-token field indices {indices} for {len(fields)} metadata fields")
            fields = [fields[index] for index in indices]

        schema = []
        for field in fields:
            labels = list(field.get("labels", []))
            if not labels:
                raise ValueError(f"Missing state-token labels for field {field.get('name')!r}: {path}")
            schema.append({"name": str(field.get("name", "state")), "labels": labels})

        expected = tuple(int(value) for value in expected_num_values)
        actual = tuple(len(field["labels"]) for field in schema)
        if actual != expected:
            raise ValueError(
                f"State-token schema mismatch for {path}: metadata category counts={actual}, model={expected}"
            )
        self.state_token_schema = schema
        self.key_state_task = (config.get("dataset") or {}).get("task")

    def encode_state_token_values(self, values):
        if not self.state_token_enabled:
            raise ValueError("state-token values require a state-token checkpoint")
        if not isinstance(values, dict):
            raise TypeError(f"oracle state-token values must be a mapping, got {type(values).__name__}")
        expected_names = [field["name"] for field in self.state_token_schema]
        missing = [name for name in expected_names if name not in values]
        extra = [name for name in values if name not in expected_names]
        if missing or extra:
            raise ValueError(f"oracle state-token fields mismatch: missing={missing}, extra={extra}")

        encoded = []
        for field in self.state_token_schema:
            value = str(values[field["name"]])
            try:
                encoded.append(field["labels"].index(value))
            except ValueError as exc:
                raise ValueError(
                    f"Unknown oracle value {value!r} for state field {field['name']!r}; "
                    f"expected one of {field['labels']}"
                ) from exc
        return np.asarray(encoded, dtype=np.int32)

    def _requires_key_state_metadata(self):
        values = [
            self.train_config_name,
            self.model_name,
            self.policy_metadata.get("key_state_variant"),
            self.policy_metadata.get("key_state_schema"),
        ]
        return any("key_state" in str(value) for value in values if value)

    def _load_key_state_config(self, path):
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        config = payload.get("config", payload)
        layout = config.get("state_layout", {})
        self.state_dim = int(layout.get("state_dim", 32))
        self.robot_dim = int(layout.get("robot_dim", 14))
        self.key_state_task = (config.get("dataset") or {}).get("task")

        schema = []
        phase = config.get("phase")
        if phase:
            schema.append(self._normalize_key_state_entry("phase", phase, "phase"))
        schema.extend(
            self._normalize_key_state_entry(execution.get("name", "execution"), execution, "execution")
            for execution in config.get("execution", [])
        )
        schema.extend(
            self._normalize_key_state_entry(attr.get("name", "attribute"), attr, "attribute")
            for attr in config.get("attributes", [])
        )

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
        encoding = entry["encoding"]
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return self._initial_key_state_index(entry)
        if encoding == "one_hot":
            return self._clip_label_index(int(np.argmax(values)), labels)
        if encoding == "label_id":
            return self._clip_label_index(int(np.rint(float(values[0]))), labels)
        raise ValueError(f"Unsupported key-state encoding: {encoding}")

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

    def set_oracle_key_state_values(self, values):
        if not self.key_state_enabled:
            raise ValueError("oracle dense key-state values require a full key-state checkpoint")
        if not isinstance(values, dict):
            raise TypeError(f"oracle key-state values must be a mapping, got {type(values).__name__}")
        expected_names = [entry["name"] for entry in self.key_state_schema]
        missing = [name for name in expected_names if name not in values]
        extra = [name for name in values if name not in expected_names]
        if missing or extra:
            raise ValueError(f"oracle key-state fields mismatch: missing={missing}, extra={extra}")
        for entry in self.key_state_schema:
            field_name = entry["name"]
            value = str(values[field_name])
            labels = entry["labels"]
            try:
                index = labels.index(value)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown oracle value {value!r} for key-state field {field_name!r}; "
                    f"expected one of {labels}"
                ) from exc
            self._write_key_state_value(entry, index)

    def uses_oracle_state(self):
        return self.state_token_rollout_mode == "oracle" or self.key_state_rollout_mode == "oracle"

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
            action_slice = action[start:end]
            state_slice = self.key_state[start:end]
            if action_slice.size == 0 or state_slice.size == 0:
                continue

            pred = self._decode_key_state_index(entry, action_slice)
            current = self._decode_key_state_index(entry, state_slice)

            if entry["kind"] == "phase":
                max_step = 1
                next_value = min(max(current, pred), current + max_step)
            else:
                labels = entry["labels"]
                unknown_index = labels.index("unknown") if "unknown" in labels else None
                if unknown_index is not None:
                    next_value = pred if current == unknown_index and pred != unknown_index else current
                else:
                    next_value = pred

            self._write_key_state_value(entry, next_value)

    def get_eval_video_overlay(self):
        if self.state_token_enabled:
            items = [
                {"label": "task", "value": self.key_state_task or self.model_name},
                {"label": "mode", "value": self.state_token_mode + "/" + self.state_token_rollout_mode},
            ]
            for field, value in zip(self.state_token_schema, self.state_token_ids, strict=True):
                index = int(value)
                labels = field["labels"]
                label = labels[index] if 0 <= index < len(labels) else f"invalid:{index}"
                items.append({"label": field["name"], "value": f"{label} [{index}]"})
            return {"title": "state-token", "items": items}
        if not self.key_state_enabled:
            return None
        items = [
            {"label": "task", "value": self.key_state_task or self.model_name},
            {"label": "update", "value": self.key_state_update_mode},
            {"label": "rollout", "value": self.key_state_rollout_mode},
        ]
        for entry in self.key_state_schema:
            start, end = entry["dim"]
            values = self.key_state[start:end]
            if values.size == 0:
                continue
            index = self._decode_key_state_index(entry, values)
            labels = entry.get("labels", [])
            value = f"{labels[index]} [{index}]" if index < len(labels) else str(index)
            items.append({"label": entry["name"], "value": value})
        return {"title": "key-state", "items": items}

    def _display_variant_name(self):
        prefix = "pi0_put_back_block_key_state_"
        if self.model_name.startswith(prefix):
            return self.model_name[len(prefix):]
        return self.model_name

    def action_for_env(self, action):
        return np.asarray(action, dtype=np.float32)[:self.robot_dim]

    def _state_for_policy(self, state):
        state = np.asarray(state, dtype=np.float32)
        if not self.key_state_enabled:
            return state
        policy_state = np.zeros(self.state_dim, dtype=np.float32)
        policy_state[:self.robot_dim] = state[:self.robot_dim]
        for entry in self.key_state_schema:
            start, end = entry["dim"]
            policy_state[start:end] = self.key_state[start:end]
        return policy_state

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr[0], img_arr[1], img_arr[2]
        policy_state = self._state_for_policy(state)
        if not self.state_history:
            self.state_history.extend(policy_state.copy() for _ in range(self.state_sequence_length))
        else:
            self.state_history.append(policy_state.copy())
        policy_state_input = (
            np.stack(self.state_history, axis=0) if self.state_history_size > 0 else self.state_history[-1]
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": policy_state_input,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self, oracle_state=None):
        assert self.observation_window is not None, "update observation_window first!"
        key_state_override = None
        if self.state_token_rollout_mode == "oracle":
            if oracle_state is None:
                raise ValueError("oracle state-token rollout requires current environment state")
            key_state_override = self.encode_state_token_values(oracle_state)
        outputs = self.policy.infer(self.observation_window, key_state_override=key_state_override)
        if self.state_token_enabled:
            state_token_ids = np.asarray(outputs["key_state"], dtype=np.int32)
            predicted_ids = np.asarray(outputs["key_state_prediction"], dtype=np.int32)
            expected_shape = (len(self.state_token_schema),)
            if state_token_ids.shape != expected_shape or predicted_ids.shape != expected_shape:
                raise ValueError(
                    "Expected state-token IDs and predictions with shape "
                    f"{expected_shape}, got executed={state_token_ids.shape}, predicted={predicted_ids.shape}"
                )
            self.state_token_ids = state_token_ids
            self.state_token_prediction_ids = predicted_ids
        return outputs["actions"]

    def get_state_token_diagnostics(self):
        if not self.state_token_enabled:
            return None
        return {
            "rollout_mode": self.state_token_rollout_mode,
            "executed_ids": self.state_token_ids.tolist(),
            "predicted_ids": self.state_token_prediction_ids.tolist(),
        }

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self.state_history.clear()
        self.policy.reset()
        self.reset_key_state()
        print("successfully unset obs and language intruction")
