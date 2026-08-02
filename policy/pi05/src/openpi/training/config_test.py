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
