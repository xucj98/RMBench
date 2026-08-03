import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Number of proprioceptive state frames: history + current + future.
    state_sequence_length: int = 1
    # Keep the current Pi0.5 state in the discrete prefix and additionally condition the action expert on the sequence.
    pi05_state_sequence_in_suffix: bool = False
    # Index of the current state inside the sequence.
    state_sequence_current_index: int | None = None

    # Opt-in structured key-state token extension. Disabled preserves the exact
    # legacy parameter tree and execution path.
    key_state_token_mode: str = "disabled"
    key_state_num_values: tuple[int, ...] = (3, 3, 3)
    key_state_loss_weight: float = 0.1

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pi05_state_sequence_in_suffix and not self.pi05:
            raise ValueError("pi05_state_sequence_in_suffix requires pi05=True")
        if self.key_state_token_mode not in {"disabled", "parallel", "serial"}:
            raise ValueError(
                "key_state_token_mode must be one of disabled, parallel, serial; " f"got {self.key_state_token_mode!r}"
            )
        if self.key_state_token_mode != "disabled" and not self.pi05:
            raise ValueError("key-state tokens are currently supported only for pi05=True")
        if self.key_state_token_mode != "disabled" and not self.key_state_num_values:
            raise ValueError("key_state_num_values must contain at least one field")
        if any(size <= 0 for size in self.key_state_num_values):
            raise ValueError("all key_state_num_values entries must be positive")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct(
                    [batch_size, self.state_sequence_length, self.action_dim]
                    if self.state_sequence_length > 1
                    else [batch_size, self.action_dim],
                    jnp.float32,
                ),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                key_state_input_ids=(
                    jax.ShapeDtypeStruct([batch_size, len(self.key_state_num_values)], jnp.int32)
                    if self.key_state_token_mode != "disabled"
                    else None
                ),
                key_state_target_ids=(
                    jax.ShapeDtypeStruct([batch_size, len(self.key_state_num_values)], jnp.int32)
                    if self.key_state_token_mode != "disabled"
                    else None
                ),
                key_state_target_mask=(
                    jax.ShapeDtypeStruct([batch_size, len(self.key_state_num_values)], jnp.bool_)
                    if self.key_state_token_mode != "disabled"
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
