import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Aloha policy."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


@dataclasses.dataclass(frozen=True)
class KeyStateTokenAlohaInputs(transforms.DataTransformFn):
    """ALOHA inputs with discrete key-state sidecars and optional hard guard targets."""

    adapt_to_pi: bool = True
    hard_action_boundary: bool = False
    key_state_field_indices: tuple[int, ...] | None = None
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = AlohaInputs.EXPECTED_CAMERAS

    def _select_key_state_fields(self, values, *, name: str, dtype) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        # Hugging Face datasets scalarize features declared with shape (1,).
        # Restore the field axis before batching so a phase-only token remains [B, 1].
        if array.ndim == 0:
            array = array.reshape(1)
        if self.key_state_field_indices is None:
            return array
        indices = tuple(self.key_state_field_indices)
        if array.shape[-1] == len(indices):
            return array
        if not indices or min(indices) < 0 or max(indices) >= array.shape[-1]:
            raise ValueError(f"{name} has {array.shape[-1]} fields, cannot select key-state indices {indices}")
        return array[..., list(indices)]

    def __call__(self, data: dict) -> dict:
        # Reuse the standard robot/image conversion; key-state values never enter
        # continuous state or action normalization.
        robot_data = {**data, "state": np.asarray(data["state"])[..., :14]}
        if "actions" in data:
            robot_data["actions"] = np.asarray(data["actions"])[..., :14]
        standard = AlohaInputs(adapt_to_pi=self.adapt_to_pi)(robot_data)
        default_num_fields = 3 if self.key_state_field_indices is None else len(self.key_state_field_indices)
        standard["key_state_input_ids"] = self._select_key_state_fields(
            data.get("key_state_input_ids", np.zeros(default_num_fields, dtype=np.int32)),
            name="key_state_input_ids",
            dtype=np.int32,
        )
        if "actions" in data:
            for key in ("key_state_target_ids", "key_state_target_mask"):
                if key not in data:
                    raise ValueError(f"structured key-state dataset is missing {key}")
            standard["key_state_target_ids"] = self._select_key_state_fields(
                data["key_state_target_ids"], name="key_state_target_ids", dtype=np.int32
            )
            standard["key_state_target_mask"] = self._select_key_state_fields(
                data["key_state_target_mask"], name="key_state_target_mask", dtype=np.bool_
            )

        if self.hard_action_boundary and "actions" in standard:
            if "key_state_guard_offset" not in data:
                raise ValueError("hard action boundary requires key_state_guard_offset")
            offset = int(np.asarray(data["key_state_guard_offset"]).reshape(-1)[0])
            actions = np.array(standard["actions"], copy=True)
            if 0 < offset < actions.shape[0]:
                actions[offset:] = actions[offset - 1]
            standard["actions"] = actions
        return standard


@dataclasses.dataclass(frozen=True)
class KeyStateAlohaInputs(transforms.DataTransformFn):
    """Aloha inputs that preserve key-state dimensions after the first 14 robot dims."""

    adapt_to_pi: bool = True

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = AlohaInputs.EXPECTED_CAMERAS

    def __call__(self, data: dict) -> dict:
        robot_state, key_state = _split_robot_key_state(np.asarray(data["state"]))
        data = {**data, "state": robot_state}
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)
        data["state"] = _join_robot_key_state(data["state"], key_state)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]
        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        if "actions" in data:
            robot_actions, key_actions = _split_robot_key_state(np.asarray(data["actions"]))
            robot_actions = _encode_actions_inv(robot_actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = _join_robot_key_state(robot_actions, key_actions)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class KeyStateAlohaOutputs(transforms.DataTransformFn):
    """Aloha outputs that return robot actions plus key-state predictions."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        robot_actions, key_actions = _split_robot_key_state(np.asarray(data["actions"]))
        robot_actions = _encode_actions(robot_actions, adapt_to_pi=self.adapt_to_pi)
        return {"actions": _join_robot_key_state(robot_actions, key_actions)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[..., [6, 13]] = _gripper_to_angular(state[..., [6, 13]])
    return state


def _split_robot_key_state(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x[..., :14], x[..., 14:]


def _join_robot_key_state(robot: np.ndarray, key_state: np.ndarray) -> np.ndarray:
    if key_state.shape[-1] == 0:
        return robot
    return np.concatenate([robot, key_state], axis=-1)


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
