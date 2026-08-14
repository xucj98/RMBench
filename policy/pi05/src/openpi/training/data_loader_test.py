import dataclasses
from typing import ClassVar

import jax
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_dataset_requests_proprioceptive_history(monkeypatch):
    captured = {}

    class Metadata:
        fps = 50
        tasks: ClassVar[dict] = {}

    class Dataset:
        def __init__(self, repo_id, **kwargs):
            captured["repo_id"] = repo_id
            captured.update(kwargs)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda _: Metadata())
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", Dataset)
    data_config = _config.DataConfig(
        repo_id="history_repo",
        action_sequence_keys=("action",),
        state_sequence_key="observation.state",
        state_history_size=3,
        state_future_size=0,
        state_step=1,
    )

    _data_loader.create_torch_dataset(data_config, 50, pi0_config.Pi0Config())

    assert captured["delta_timestamps"]["action"] == [step / 50 for step in range(50)]
    assert captured["delta_timestamps"]["observation.state"] == [-3 / 50, -2 / 50, -1 / 50, 0]


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _KeyStateTable:
    column_names: ClassVar = ["observation.key_state_input_ids", "observation.key_state_target_ids"]

    def __init__(self, rows):
        self._rows = rows

    def select_columns(self, _columns):
        return self

    def __getitem__(self, index):
        return self._rows[index]


class _KeyStateDataset:
    def __init__(self, size=100):
        self._rows = [
            {
                "frame_index": np.asarray(index),
                "observation.key_state_input_ids": np.asarray([0, 0]),
                "observation.key_state_target_ids": np.asarray([index, 1]),
            }
            for index in range(size)
        ]
        self.hf_dataset = _KeyStateTable(self._rows)

    def __getitem__(self, index):
        return dict(self._rows[index])

    def __len__(self):
        return len(self._rows)


def test_random_previous_key_state_uses_target_at_sampled_lag_and_episode_initial_state():
    dataset = _data_loader.RandomPreviousKeyStateDataset(_KeyStateDataset(), (2, 2))

    np.testing.assert_array_equal(dataset[4]["observation.key_state_input_ids"], [2, 1])
    np.testing.assert_array_equal(dataset[1]["observation.key_state_input_ids"], [0, 0])


def test_random_previous_key_state_resamples_inclusive_lag_range():
    dataset = _data_loader.RandomPreviousKeyStateDataset(_KeyStateDataset(), (15, 50))
    torch.manual_seed(0)

    sampled_lags = {80 - int(dataset[80]["observation.key_state_input_ids"][0]) for _ in range(20)}

    assert sampled_lags <= set(range(15, 51))
    assert len(sampled_lags) > 1
