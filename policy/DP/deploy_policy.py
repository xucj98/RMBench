import numpy as np
from .dp_model import DP
from pathlib import Path


def encode_obs(observation, model=None):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    state = observation["joint_action"]["vector"]
    if model is not None and hasattr(model, "state_for_policy"):
        state = model.state_for_policy(state)
    obs["agent_pos"] = state
    return obs


def resolve_checkpoint_run_dir(usr_args):
    checkpoint_task_config = usr_args.get("checkpoint_task_config") or usr_args["task_config"]
    return Path(
        f"./policy/DP/checkpoints/"
        f"{usr_args['task_name']}-{checkpoint_task_config}-{usr_args['expert_data_num']}-{usr_args['seed']}"
    )


def get_model(usr_args):
    checkpoint_run_dir = resolve_checkpoint_run_dir(usr_args)
    ckpt_file = checkpoint_run_dir / f"{usr_args['checkpoint_num']}.ckpt"

    ddim_steps = usr_args.get('ddim_steps', None)
    key_state_update_mode = usr_args.get("key_state_update_mode", "raw")
    key_state_config_path = checkpoint_run_dir / "metadata" / "rmbench_data_meta" / "key_state_config.yaml"

    return DP(
        str(ckpt_file),
        ddim_steps=ddim_steps,
        key_state_config_path=key_state_config_path,
        key_state_update_mode=key_state_update_mode,
    )


def sync_eval_video_overlay(TASK_ENV, model):
    if hasattr(TASK_ENV, "set_eval_video_overlay") and hasattr(model, "get_eval_video_overlay"):
        TASK_ENV.set_eval_video_overlay(model.get_eval_video_overlay())


def should_capture_observation(action_index, action_count, n_obs_steps, recording_video):
    """Return whether an action's post-observation is needed before the next inference."""
    if action_index >= action_count - 1:
        # The outer eval loop captures the final post-action observation.
        return False
    if recording_video:
        return True
    first_required_index = max(0, action_count - n_obs_steps)
    return action_index >= first_required_index


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation, model)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    actions = model.get_action(obs)

    action_count = len(actions)
    recording_video = TASK_ENV.is_recording_eval_video()
    for action_index, action in enumerate(actions):
        sync_eval_video_overlay(TASK_ENV, model)
        TASK_ENV.take_action(model.action_for_env(action))
        model.update_key_state_from_action(action)
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        if should_capture_observation(
            action_index,
            action_count,
            model.n_obs_steps,
            recording_video,
        ):
            observation = TASK_ENV.get_obs()
            model.update_obs(encode_obs(observation, model))


def reset_model(model):
    model.reset_obs()
