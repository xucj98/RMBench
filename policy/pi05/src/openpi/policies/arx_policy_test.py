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


def test_legacy_sm2sm_scalar_inference_keeps_warm_start_inpainting_sidecar():
    state = np.zeros(32, dtype=np.float32)
    state[31] = 1.0
    transform = arx_policy.ArxSm2smInputs(representation="full_state")

    result = transform({"state": state, "images": _images()})

    assert result["state_inpainting_mask"] == 1.0


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
    expected_robot_mask = np.repeat(np.asarray([[True], [False], [False], [False]]), 28, axis=1)
    np.testing.assert_array_equal(mask[:, :28], expected_robot_mask)
    expected_memory_mask = np.repeat(np.asarray([[True], [False], [False], [True]]), 3, axis=1)
    np.testing.assert_array_equal(mask[:, 28:31], expected_memory_mask)
    assert not mask[:, 31].any()


def test_forced_current_execution_keeps_robot_actions_until_next_unknown_execution():
    transform = arx_policy.ArxSm2smInputs(representation="state_token")
    result = transform(
        {
            "state": np.zeros(32, dtype=np.float32),
            "actions": np.zeros((5, 32), dtype=np.float32),
            "memory_action_valid": np.asarray([[False], [False], [True], [True], [False]]),
            "key_state_input_ids": np.asarray([2]),
            "key_state_target_ids": np.asarray([2]),
            "key_state_target_mask": np.asarray([True]),
            "images": _images(),
        }
    )

    mask = result["action_loss_mask"]
    expected_robot_mask = np.repeat(np.asarray([[True], [True], [True], [True], [False]]), 28, axis=1)
    np.testing.assert_array_equal(mask[:, :28], expected_robot_mask)
    assert not mask[:, 28:].any()


def test_s2m_full_state_maps_each_memory_field_mask_and_has_no_inpainting_sidecar():
    transform = arx_policy.ArxSm2smInputs(
        representation="full_state",
        robot_state_dim=14,
        memory_dim=6,
        memory_field_dims=(3, 3),
        state_history_size=0,
        state_future_size=0,
    )
    result = transform(
        {
            "state": np.zeros(32, dtype=np.float32),
            "actions": np.zeros((4, 32), dtype=np.float32),
            "memory_action_valid": np.asarray(
                [[True, True], [True, False], [True, False], [True, True]], dtype=np.bool_
            ),
            "images": _images(),
        }
    )

    assert result["state"].shape == (20,)
    assert result["actions"].shape == (4, 20)
    assert "state_inpainting_mask" not in result
    mask = result["action_loss_mask"]
    np.testing.assert_array_equal(mask[:, :14], np.repeat([[True], [False], [False], [False]], 14, axis=1))
    assert mask[:, 14:17].all()
    np.testing.assert_array_equal(mask[:, 17:20], np.repeat([[True], [False], [False], [True]], 3, axis=1))
    assert not mask[:, 20:].any()


def test_s2m_serial_state_token_uses_two_fields_and_master_only_actions():
    transform = arx_policy.ArxSm2smInputs(
        representation="state_token",
        robot_state_dim=14,
        memory_dim=6,
        memory_field_dims=(3, 3),
        state_history_size=0,
        state_future_size=0,
    )
    result = transform(
        {
            "state": np.zeros(32, dtype=np.float32),
            "actions": np.zeros((2, 32), dtype=np.float32),
            "memory_action_valid": np.ones((2, 2), dtype=np.bool_),
            "key_state_input_ids": np.asarray([1, 3]),
            "key_state_target_ids": np.asarray([2, 0]),
            "key_state_target_mask": np.asarray([True, True]),
            "images": _images(),
        }
    )

    assert result["state"].shape == (14,)
    assert result["actions"].shape == (2, 14)
    np.testing.assert_array_equal(result["key_state_input_ids"], [1, 3])
    np.testing.assert_array_equal(result["key_state_target_ids"], [2, 0])
