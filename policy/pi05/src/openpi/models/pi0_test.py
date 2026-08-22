import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _dummy_state_token_config(mode: str, key_state_num_values: tuple[int, ...] = (3, 3, 3)) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=14,
        max_token_len=8,
        key_state_token_mode=mode,
        key_state_num_values=key_state_num_values,
    )


def test_disabled_config_has_no_key_state_params():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    state = nnx.state(config.create(jax.random.key(0))).flat_state()
    assert all("key_state_token" not in "/".join(path) for path in state)


def test_action_loss_mask_can_remove_all_continuous_supervision():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=14,
        action_horizon=4,
        max_token_len=8,
        use_action_loss_mask=True,
    )
    model = config.create(jax.random.key(0))
    observation = dataclasses.replace(
        config.fake_obs(batch_size=2),
        action_loss_mask=jnp.zeros((2, 4, 14), dtype=jnp.bool_),
    )

    loss = model.compute_loss(jax.random.key(1), observation, config.fake_act(batch_size=2))

    assert not loss.any()


@pytest.mark.parametrize("mode", ["parallel", "serial"])
def test_key_state_token_loss_and_rollout(mode):
    config = _dummy_state_token_config(mode)
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=2)
    actions = config.fake_act(batch_size=2)

    loss = model.compute_loss(jax.random.key(1), observation, actions)
    sampled, state_ids, logits = model.sample_actions_with_key_state(jax.random.key(2), observation, num_steps=2)

    assert loss.shape == (2, config.action_horizon)
    assert sampled.shape == (2, config.action_horizon, config.action_dim)
    assert state_ids.shape == (2, 3)
    assert logits.shape == (2, 3, 3)


def test_two_field_serial_state_token_loss_and_rollout():
    config = _dummy_state_token_config("serial", (3, 3))
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=2)
    actions = config.fake_act(batch_size=2)

    loss = model.compute_loss(jax.random.key(1), observation, actions)
    sampled, state_ids, logits = model.sample_actions_with_key_state(jax.random.key(2), observation, num_steps=2)

    assert loss.shape == (2, config.action_horizon)
    assert sampled.shape == (2, config.action_horizon, config.action_dim)
    assert state_ids.shape == (2, 2)
    assert logits.shape == (2, 2, 3)


def test_four_field_serial_state_token_loss_and_rollout():
    config = _dummy_state_token_config("serial", (6, 4, 4, 4))
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=2)
    actions = config.fake_act(batch_size=2)

    loss = model.compute_loss(jax.random.key(1), observation, actions)
    sampled, state_ids, logits = model.sample_actions_with_key_state(jax.random.key(2), observation, num_steps=2)

    assert loss.shape == (2, config.action_horizon)
    assert sampled.shape == (2, config.action_horizon, config.action_dim)
    assert state_ids.shape == (2, 4)
    assert logits.shape == (2, 4, 6)


def test_multitask_phase_and_attribute_transition_mask():
    model = _dummy_state_token_config("serial", (6, 4, 4, 4)).create(jax.random.key(0))
    logits = jnp.asarray(
        [
            [
                [0.0, 5.0, 100.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 100.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 100.0, 0.0, 0.0],
                [0.0, 100.0, 2.0, 3.0, 0.0, 0.0],
            ]
        ]
    )
    previous = jnp.asarray([[0, 2, 0, 3]], dtype=jnp.int32)

    # Phase cannot skip; known attributes stay latched; unknown may resolve.
    assert model._select_key_state(logits, previous).tolist() == [[1, 2, 3, 3]]  # noqa: SLF001


def test_two_field_key_state_transition_mask():
    model = _dummy_state_token_config("serial", (3, 3)).create(jax.random.key(0))
    logits = jnp.asarray([[[0.0, 2.0, 100.0], [0.0, 3.0, 2.0]]])
    previous = jnp.asarray([[0, 0]], dtype=jnp.int32)
    assert model._select_key_state(logits, previous).tolist() == [[1, 1]]  # noqa: SLF001


def test_key_state_transition_mask():
    model = _dummy_state_token_config("parallel").create(jax.random.key(0))
    logits = jnp.asarray([[[0.0, 2.0, 100.0], [0.0, 3.0, 2.0], [100.0, 2.0, 1.0]]])
    previous = jnp.asarray([[0, 0, 0]], dtype=jnp.int32)
    # P0 cannot skip to P2; entering P1 forces button=unconfirmed.
    assert model._select_key_state(logits, previous).tolist() == [[1, 1, 1]]  # noqa: SLF001


def test_serial_state_token_accepts_oracle_action_condition():
    predicted_ids = jnp.asarray([[0, 0, 0], [1, 2, 1]], dtype=jnp.int32)
    oracle_ids = jnp.asarray([[0, 1, 0], [2, 2, 0]], dtype=jnp.int32)

    condition_ids = _pi0.Pi0._resolve_action_condition_state_ids(  # noqa: SLF001
        predicted_ids, oracle_ids
    )

    assert condition_ids.tolist() == oracle_ids.tolist()


def test_serial_state_token_defaults_to_prediction_without_oracle():
    predicted_ids = jnp.asarray([[0, 1, 0]], dtype=jnp.int32)

    condition_ids = _pi0.Pi0._resolve_action_condition_state_ids(  # noqa: SLF001
        predicted_ids, None
    )

    assert condition_ids is predicted_ids


def test_serial_state_token_rejects_wrong_oracle_shape():
    predicted_ids = jnp.zeros((2, 3), dtype=jnp.int32)
    with pytest.raises(ValueError, match="must match predicted state shape"):
        _pi0.Pi0._resolve_action_condition_state_ids(  # noqa: SLF001
            predicted_ids, jnp.zeros((2, 2), dtype=jnp.int32)
        )
