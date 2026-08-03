import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def test_key_state_token_hard_boundary_repeats_last_pre_guard_action():
    actions = np.arange(6 * 14, dtype=np.float32).reshape(6, 14)
    data = {
        "state": np.zeros(14, dtype=np.float32),
        "actions": actions,
        "images": {"cam_high": np.zeros((3, 8, 8), dtype=np.uint8)},
        "key_state_input_ids": np.asarray([0, 0, 0]),
        "key_state_target_ids": np.asarray([1, 1, 1]),
        "key_state_target_mask": np.ones(3, dtype=np.bool_),
        "key_state_guard_offset": np.asarray([3]),
    }
    transformed = aloha_policy.KeyStateTokenAlohaInputs(adapt_to_pi=False, hard_action_boundary=True)(data)
    np.testing.assert_array_equal(transformed["actions"][:3], actions[:3])
    np.testing.assert_array_equal(transformed["actions"][3:], np.repeat(actions[2:3], 3, axis=0))


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
