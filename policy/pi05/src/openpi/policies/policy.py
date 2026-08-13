from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            self._key_state_token_enabled = getattr(model, "key_state_token_mode", "disabled") != "disabled"
            if self._key_state_token_enabled:
                self._sample_actions_with_key_state = nnx_utils.module_jit(model.sample_actions_with_key_state)
                self._key_state_num_values = tuple(int(value) for value in getattr(model, "key_state_num_values", ()))
                self._key_state_num_fields = len(self._key_state_num_values)
                if self._key_state_num_fields <= 0:
                    raise ValueError("key-state token model must define at least one field")
                self._key_state_previous = np.zeros(self._key_state_num_fields, dtype=np.int32)

    @override
    def reset(self) -> None:
        if getattr(self, "_key_state_token_enabled", False):
            self._key_state_previous = np.zeros(self._key_state_num_fields, dtype=np.int32)

    def _validate_key_state_override(self, value: np.ndarray) -> np.ndarray:
        override = np.asarray(value, dtype=np.int32)
        if override.shape != (self._key_state_num_fields,):
            raise ValueError(
                "key_state_override must have one value per state field: "
                f"expected {(self._key_state_num_fields,)}, got {override.shape}"
            )
        for field_index, (field_value, num_values) in enumerate(
            zip(override, self._key_state_num_values, strict=True)
        ):
            if not 0 <= int(field_value) < num_values:
                raise ValueError(
                    f"key_state_override field {field_index} is out of range: "
                    f"value={int(field_value)}, num_values={num_values}"
                )
        return override

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        key_state_override: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if key_state_override is not None:
            if not getattr(self, "_key_state_token_enabled", False):
                raise ValueError("key_state_override requires a key-state token model")
            key_state_override = self._validate_key_state_override(key_state_override)
        if getattr(self, "_key_state_token_enabled", False):
            inputs["key_state_input_ids"] = self._key_state_previous
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if key_state_override is not None:
            sample_kwargs["action_condition_state_ids"] = jnp.asarray(key_state_override)[None, ...]
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        key_state_diagnostics = None
        if getattr(self, "_key_state_token_enabled", False):
            actions, key_state_ids, key_state_logits = self._sample_actions_with_key_state(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )
            outputs = {"state": inputs["state"], "actions": actions}
            key_state_diagnostics = (key_state_ids, key_state_logits)
        else:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if key_state_diagnostics is not None:
            key_state_ids, key_state_logits = key_state_diagnostics
            predicted_ids = np.asarray(key_state_ids[0, ...], dtype=np.int32)
            executed_ids = predicted_ids if key_state_override is None else key_state_override
            self._key_state_previous = np.asarray(executed_ids, dtype=np.int32)
            outputs["key_state"] = self._key_state_previous.copy()
            outputs["key_state_prediction"] = predicted_ids
            outputs["key_state_logits"] = np.asarray(key_state_logits[0, ...])
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
