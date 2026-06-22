"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os
import shlex
import subprocess

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def get_git_status():
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=pathlib.Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
    return "clean" if not status else status


def get_runtime_env():
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "HYDRA_FULL_ERROR",
        "PYTHONPATH",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "WANDB_MODE",
    ]
    return {key: os.environ[key] for key in keys if key in os.environ}


def attach_runtime_config(cfg: OmegaConf):
    OmegaConf.set_struct(cfg, False)
    cfg["_runtime"] = {
        "git_commit": get_git_commit(),
        "git_status": get_git_status(),
        "cwd": str(pathlib.Path.cwd()),
        "command": " ".join(shlex.quote(item) for item in sys.argv),
        "env": get_runtime_env(),
    }


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]
    OmegaConf.resolve(cfg)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    cfg.task.shape_meta.obs.head_cam.shape = [
        3,
        head_camera_cfg["h"],
        head_camera_cfg["w"],
    ]
    attach_runtime_config(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
