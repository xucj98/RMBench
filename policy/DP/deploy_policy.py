import numpy as np
from .dp_model import DP
import yaml
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
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper

    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']

    ddim_steps = usr_args.get('ddim_steps', None)
    key_state_update_mode = usr_args.get("key_state_update_mode", "raw")
    key_state_config_path = checkpoint_run_dir / "metadata" / "rmbench_data_meta" / "key_state_config.yaml"

    return DP(
        str(ckpt_file),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        ddim_steps=ddim_steps,
        key_state_config_path=key_state_config_path,
        key_state_update_mode=key_state_update_mode,
    )


def sync_eval_video_overlay(TASK_ENV, model):
    if hasattr(TASK_ENV, "set_eval_video_overlay") and hasattr(model, "get_eval_video_overlay"):
        TASK_ENV.set_eval_video_overlay(model.get_eval_video_overlay())


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

    for action in actions:
        sync_eval_video_overlay(TASK_ENV, model)
        TASK_ENV.take_action(model.action_for_env(action))
        model.update_key_state_from_action(action)
        observation = TASK_ENV.get_obs_for_policy()
        obs = encode_obs(observation, model)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
