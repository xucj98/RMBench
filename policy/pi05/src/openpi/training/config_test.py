from openpi.training import config as _config


def test_pi05_prop_history_config():
    config = _config.get_config("pi05_full_key_state_with_prop_history")

    assert config.num_train_steps == 30_000
    assert config.batch_size == 32
    assert config.data.state_history_size == 3
    assert config.data.state_future_size == 0
    assert config.model.state_sequence_length == 4
    assert config.model.state_sequence_current_index == 3
    assert config.model.pi05_state_sequence_in_suffix


def test_rearrange_state_token_ablation_uses_one_shared_config():
    config = _config.get_config("pi05_rearrange_state_token_boundary_ablation")

    assert config.batch_size == 32
    assert config.num_train_steps == 30_000
    assert config.model.key_state_token_mode == "parallel"
    assert not config.data.hard_action_boundary
    assert config.data.assets.asset_id == "rearrange_blocks_state_token"
    assert config.policy_metadata["batch_id"] == config.name
    assert config.policy_metadata["serial_train_conditioning"] == "teacher_forcing"


def test_rearrange_state_token_no_button_ablation_is_strict_single_factor():
    full = _config.get_config("pi05_rearrange_state_token_boundary_ablation")
    config = _config.get_config("pi05_rearrange_state_token_no_button_ablation")

    assert config.batch_size == full.batch_size == 32
    assert config.num_train_steps == full.num_train_steps == 30_000
    assert config.model.key_state_token_mode == "serial"
    assert config.model.key_state_num_values == (3, 3)
    assert config.data.repo_id == full.data.repo_id
    assert config.data.assets.asset_id == full.data.assets.asset_id
    assert config.data.key_state_field_indices == (0, 1)
    assert not config.data.hard_action_boundary
    assert config.policy_metadata["batch_id"] == full.name
    assert config.policy_metadata["ablation"] == "remove_button_press_status"
