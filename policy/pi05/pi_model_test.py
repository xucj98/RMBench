from collections import deque

import numpy as np
import pytest
import yaml

from openpi.training import config as _config
from pi_model import PI0


class _ResettablePolicy:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_state_token_metadata_is_loaded_without_legacy_state_layout(tmp_path):
    metadata_path = tmp_path / "key_state_config.yaml"
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"task": "rearrange_blocks"},
                "state_layout": {"state_dim": 14, "robot_dim": 14},
                "structured_state_tokens": {
                    "fields": [
                        {"name": "phase", "labels": ["P0", "P1", "P2"]},
                        {"name": "side", "labels": ["unknown", "left", "right"]},
                        {"name": "button", "labels": ["NA", "unconfirmed", "confirmed"]},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    model = PI0.__new__(PI0)

    model._load_state_token_config(metadata_path, (3, 3, 3))  # noqa: SLF001

    assert model.key_state_task == "rearrange_blocks"
    assert [field["name"] for field in model.state_token_schema] == ["phase", "side", "button"]


def test_state_token_metadata_category_mismatch_fails_fast(tmp_path):
    metadata_path = tmp_path / "key_state_config.yaml"
    metadata_path.write_text(
        yaml.safe_dump(
            {
                "structured_state_tokens": {
                    "fields": [{"name": "phase", "labels": ["P0", "P1"]}]
                }
            }
        ),
        encoding="utf-8",
    )
    model = PI0.__new__(PI0)

    with pytest.raises(ValueError, match="schema mismatch"):
        model._load_state_token_config(metadata_path, (3,))  # noqa: SLF001


def test_episode_reset_reaches_state_token_policy_memory():
    model = PI0.__new__(PI0)
    model.instruction = "stale"
    model.observation_window = {"state": np.ones(14)}
    model.state_history = deque([np.ones(14)])
    model.policy = _ResettablePolicy()
    model.state_token_enabled = True
    model.state_token_schema = [{"name": "phase", "labels": ["P0", "P1", "P2"]}]
    model.state_token_ids = np.asarray([2], dtype=np.int32)
    model.key_state_enabled = False

    model.reset_obsrvationwindows()

    assert model.policy.reset_calls == 1
    np.testing.assert_array_equal(model.state_token_ids, np.asarray([0], dtype=np.int32))
    assert model.observation_window is None
    assert not model.state_history


def test_eval_restores_state_token_mode_from_checkpoint_metadata(tmp_path):
    run_dir = tmp_path / "serial_soft_seed42"
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "train_config.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pi05_rearrange_state_token_boundary_ablation",
                "model": {
                    "key_state_token_mode": "serial",
                    "key_state_num_values": [3, 3, 3],
                },
                "policy_metadata": {"serial_train_conditioning": "teacher_forcing"},
            }
        ),
        encoding="utf-8",
    )
    model = PI0.__new__(PI0)
    model.checkpoint_run_dir = run_dir
    model.train_config_name = "pi05_rearrange_state_token_boundary_ablation"
    shared_config = _config.get_config(model.train_config_name)

    restored_config = model._load_checkpoint_train_config(shared_config)  # noqa: SLF001

    assert shared_config.model.key_state_token_mode == "parallel"
    assert restored_config.model.key_state_token_mode == "serial"
    assert restored_config.policy_metadata["serial_train_conditioning"] == "teacher_forcing"
