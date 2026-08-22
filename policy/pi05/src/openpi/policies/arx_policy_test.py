import numpy as np

from openpi.policies import arx_policy


def _images() -> dict[str, np.ndarray]:
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    return {"left_wrist_view": image, "face_view": image, "right_wrist_view": image}


def test_full_state_masks_future_slave_and_memory_but_keeps_future_master():
    state = np.arange(7 * 32, dtype=np.float32).reshape(7, 32)
    state[:, 31] = 0.0
    current = state[3].copy()
    transform = arx_policy.ArxSm2smInputs(representation="full_state", state_history_size=3, state_future_size=3)

    result = transform({"state": state, "images": _images(), "prompt": "sort"})

    assert result["state"].shape == (7, 31)
    np.testing.assert_array_equal(result["state"][4:, :14], np.repeat(current[None, :14], 3, axis=0))
    np.testing.assert_array_equal(result["state"][4:, 28:31], np.repeat(current[None, 28:31], 3, axis=0))
    np.testing.assert_array_equal(result["state"][4:, 14:28], state[4:, 14:28])


def test_post_norm_mask_uses_reserved_dimension_31():
    transform = arx_policy.AddStateInpaintingMask(action_dim=32)
    result = transform(
        {
            "state": np.zeros((7, 28), dtype=np.float32),
            "state_inpainting_mask": np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.float32),
        }
    )

    assert result["state"].shape == (7, 32)
    np.testing.assert_array_equal(result["state"][:, 31], [0, 0, 0, 0, 1, 1, 1])
    assert "state_inpainting_mask" not in result


def test_full_state_masks_only_memory_action_loss_for_forced_execution():
    transform = arx_policy.ArxSm2smInputs(representation="full_state")
    actions = np.zeros((4, 32), dtype=np.float32)
    result = transform(
        {
            "state": np.zeros(32, dtype=np.float32),
            "actions": actions,
            "memory_action_valid": np.asarray([[True], [False], [False], [True]]),
            "images": _images(),
        }
    )

    mask = result["action_loss_mask"]
    assert mask.shape == (4, 32)
    assert mask[:, :28].all()
    expected_memory_mask = np.repeat(np.asarray([[True], [False], [False], [True]]), 3, axis=1)
    np.testing.assert_array_equal(mask[:, 28:31], expected_memory_mask)
    assert not mask[:, 31].any()
