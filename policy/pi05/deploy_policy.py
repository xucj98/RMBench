import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
    )
    key_state_update_mode = usr_args.get("key_state_update_mode", "raw")
    state_token_rollout_mode = usr_args.get("state_token_rollout_mode", "predicted")
    key_state_rollout_mode = usr_args.get("key_state_rollout_mode", "predicted")
    return PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        key_state_update_mode,
        state_token_rollout_mode,
        key_state_rollout_mode,
    )


def sync_eval_video_overlay(TASK_ENV, model):
    if hasattr(TASK_ENV, "set_eval_video_overlay") and hasattr(model, "get_eval_video_overlay"):
        TASK_ENV.set_eval_video_overlay(model.get_eval_video_overlay())


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    oracle_state = None
    if model.uses_oracle_state():
        oracle_provider = getattr(TASK_ENV, "get_oracle_key_state", None)
        if oracle_provider is None:
            raise ValueError(
                f"Task {type(TASK_ENV).__name__} does not provide get_oracle_key_state()"
            )
        oracle_state = oracle_provider()
        if model.key_state_rollout_mode == "oracle":
            model.set_oracle_key_state_values(oracle_state)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action(oracle_state=oracle_state)[:model.pi0_step]
    state_token_diagnostics = model.get_state_token_diagnostics()
    if state_token_diagnostics is not None and hasattr(TASK_ENV, "_record_eval_diagnostic_event"):
        TASK_ENV._record_eval_diagnostic_event(  # noqa: SLF001
            "state_token_query", **state_token_diagnostics
        )

    for step, action in enumerate(actions):
        sync_eval_video_overlay(TASK_ENV, model)
        TASK_ENV.take_action(model.action_for_env(action))
        if model.key_state_rollout_mode == "oracle":
            model.set_oracle_key_state_values(TASK_ENV.get_oracle_key_state())
        else:
            model.update_key_state_from_action(action)
        # The next eval call appends the final post-action observation. Avoid appending it twice.
        if step + 1 < len(actions):
            observation = TASK_ENV.get_obs_for_policy()
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
