from __future__ import annotations

import dataclasses
import random
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class ArxSm2smInputs(transforms.DataTransformFn):
    """Prepare X1Pro SM2SM trajectories and delayed state sequences.

    Raw rows use the stable deployment layout ``slave[14] + master[14] +
    optional memory + availability_mask``.  Future slave and semantic-memory
    values are hidden because they are not available online; future master
    values remain as the action-inpainting condition.
    """

    representation: str = "full_state"
    state_history_size: int = 3
    state_future_size: int = 3
    slave_state_dim: int = 14
    robot_state_dim: int = 28
    memory_dim: int = 3
    availability_mask_index: int = 31
    random_drop_master: float = 0.0
    random_drop_history: float = 0.0
    random_drop_future: float = 0.0
    random_pos_offset: float = 0.0

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    )

    def __post_init__(self) -> None:
        if self.representation not in {"full_state", "state_token"}:
            raise ValueError(f"Unsupported ARX memory representation: {self.representation!r}")
        if self.robot_state_dim != self.slave_state_dim * 2:
            raise ValueError("SM2SM robot_state_dim must contain equally-sized slave and master blocks")

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        state, availability = self._prepare_state(raw_state)

        inputs = {
            "image": {
                "base_0_rgb": self._convert_image(data["images"]["face_view"]),
                "left_wrist_0_rgb": self._convert_image(data["images"]["left_wrist_view"]),
                "right_wrist_0_rgb": self._convert_image(data["images"]["right_wrist_view"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
            "state_inpainting_mask": availability,
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            output_dim = self.robot_state_dim + (self.memory_dim if self.representation == "full_state" else 0)
            if raw_actions.shape[-1] < output_dim:
                raise ValueError(f"ARX actions need at least {output_dim} dims, got {raw_actions.shape}")
            inputs["actions"] = np.array(raw_actions[..., :output_dim], copy=True)
            if self.representation == "full_state":
                if "memory_action_valid" not in data:
                    raise ValueError("Full-state ARX dataset is missing memory_action_valid")
                memory_valid = np.asarray(data["memory_action_valid"], dtype=np.bool_)
                if memory_valid.ndim > 0 and memory_valid.shape[-1] == 1:
                    memory_valid = memory_valid[..., 0]
                if memory_valid.shape != raw_actions.shape[:-1]:
                    raise ValueError(
                        f"memory_action_valid shape {memory_valid.shape} does not match actions {raw_actions.shape}"
                    )
                action_loss_mask = np.zeros((*raw_actions.shape[:-1], self.availability_mask_index + 1), dtype=np.bool_)
                action_loss_mask[..., : self.robot_state_dim] = True
                action_loss_mask[..., self.robot_state_dim : output_dim] = memory_valid[..., None]
                inputs["action_loss_mask"] = action_loss_mask

        if self.representation == "state_token":
            input_ids = np.asarray(data["key_state_input_ids"], dtype=np.int32)
            if input_ids.ndim == 0:
                input_ids = input_ids.reshape(1)
            inputs["key_state_input_ids"] = input_ids
            if "actions" in data:
                for key, dtype in (
                    ("key_state_target_ids", np.int32),
                    ("key_state_target_mask", np.bool_),
                ):
                    if key not in data:
                        raise ValueError(f"State-token ARX dataset is missing {key}")
                    value = np.asarray(data[key], dtype=dtype)
                    inputs[key] = value.reshape(1) if value.ndim == 0 else value

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.random_pos_offset > 0.0 and "actions" in inputs:
            offset = (np.random.rand(3).astype(np.float32) * 2.0 - 1.0) * self.random_pos_offset
            inputs["state"][..., 7:10] += offset
            inputs["state"][..., 21:24] += offset
            inputs["actions"][..., 7:10] += offset
            inputs["actions"][..., 21:24] += offset

        return inputs

    def _prepare_state(self, raw_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if raw_state.ndim not in {1, 2}:
            raise ValueError(f"ARX state must be a vector or sequence, got {raw_state.shape}")
        required_dim = self.robot_state_dim + (self.memory_dim if self.representation == "full_state" else 0)
        if raw_state.shape[-1] < required_dim:
            raise ValueError(f"ARX state needs at least {required_dim} dims, got {raw_state.shape}")

        output = np.array(raw_state[..., :required_dim], copy=True)
        if raw_state.ndim == 1:
            availability = np.asarray(
                raw_state[self.availability_mask_index] if raw_state.shape[-1] > self.availability_mask_index else 0.0,
                dtype=np.float32,
            )
            return output, availability

        expected_length = self.state_history_size + 1 + self.state_future_size
        if raw_state.shape[0] != expected_length:
            raise ValueError(f"Expected {expected_length} state frames, got {raw_state.shape[0]}")
        current_index = self.state_history_size
        current_state = np.array(output[current_index], copy=True)
        current_slave = current_state[: self.slave_state_dim]

        availability = (
            np.array(raw_state[:, self.availability_mask_index], copy=True)
            if raw_state.shape[-1] > self.availability_mask_index
            else np.zeros(expected_length, dtype=np.float32)
        )
        if self.state_future_size > 0:
            future = slice(current_index + 1, None)
            output[future, : self.slave_state_dim] = current_slave
            if self.representation == "full_state":
                output[future, self.robot_state_dim : required_dim] = current_state[self.robot_state_dim : required_dim]

        if random.random() < self.random_drop_master:
            output[:, self.slave_state_dim : self.robot_state_dim] = current_slave
            availability[:] = 1.0
        if self.state_history_size > 0 and random.random() < self.random_drop_history:
            output[:current_index] = current_state
            availability[:current_index] = 1.0
        if self.state_future_size > 0 and random.random() < self.random_drop_future:
            drop_size = random.randint(1, self.state_future_size)
            output[-drop_size:] = output[-drop_size - 1]
            availability[-drop_size:] = 1.0

        return output, availability

    @staticmethod
    def _convert_image(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * image).astype(np.uint8)
        if image.shape[-1] != 3:
            image = einops.rearrange(image, "c h w -> h w c")
        if image.shape[-1] != 3:
            raise ValueError(f"Expected an RGB image, got {image.shape}")
        return image


@dataclasses.dataclass(frozen=True)
class AddStateInpaintingMask(transforms.DataTransformFn):
    """Append the availability mask after normalization and remove its sidecar."""

    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(np.asarray(data["state"]), self.action_dim)
        mask = np.asarray(data.pop("state_inpainting_mask"), dtype=state.dtype)
        if mask.shape != state.shape[:-1]:
            raise ValueError(f"Inpainting mask shape {mask.shape} does not match state {state.shape}")
        state[..., -1] = mask
        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class ArxSm2smOutputs(transforms.DataTransformFn):
    """Return the deployable SM2SM block and optional dense memory prediction."""

    representation: str = "full_state"
    robot_state_dim: int = 28
    memory_dim: int = 3

    def __call__(self, data: dict) -> dict:
        output_dim = self.robot_state_dim + (self.memory_dim if self.representation == "full_state" else 0)
        return {"actions": np.asarray(data["actions"])[..., :output_dim]}
