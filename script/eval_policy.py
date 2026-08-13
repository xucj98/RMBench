import sys
import os
import subprocess
import shutil

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import shlex
import logging

from generate_episode_instructions import *
from eval_diagnostics import EvalDiagnosticsRecorder

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def get_runtime_env():
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "SAPIEN_RENDER_DEVICE",
        "PYTHONPATH",
        "WANDB_PROJECT",
        "WANDB_RUN_GROUP",
        "WANDB_MODE",
    ]
    return {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None}


def to_yaml_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_yaml_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_eval_save_dir(usr_args, task_name, policy_name, task_config, ckpt_setting, current_time):
    eval_output_dir = usr_args.get("eval_output_dir")
    if eval_output_dir:
        return Path(str(eval_output_dir))
    return Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")


def requires_key_state_metadata(usr_args):
    values = [
        usr_args.get("train_config_name"),
        usr_args.get("model_name"),
        usr_args.get("task_config"),
        usr_args.get("checkpoint_task_config"),
        usr_args.get("ckpt_setting"),
    ]
    return any("key_state" in str(value) for value in values if value is not None)


def resolve_checkpoint_paths(usr_args):
    if usr_args.get("policy_name") == "DP":
        task_name = usr_args.get("task_name")
        task_config = usr_args.get("checkpoint_task_config") or usr_args.get("task_config")
        expert_data_num = usr_args.get("expert_data_num")
        seed = usr_args.get("seed")
        checkpoint_num = usr_args.get("checkpoint_num")
        if not task_name or not task_config or expert_data_num is None or seed is None or checkpoint_num is None:
            return {}

        checkpoint_run_dir = (
            Path("policy/DP/checkpoints")
            / f"{task_name}-{task_config}-{expert_data_num}-{seed}"
        )
        checkpoint_dir = checkpoint_run_dir / f"{checkpoint_num}.ckpt"
        checkpoint_metadata_dir = checkpoint_run_dir / "metadata"
        key_state_config = checkpoint_metadata_dir / "rmbench_data_meta" / "key_state_config.yaml"
        return {
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_run_dir": checkpoint_run_dir,
            "checkpoint_metadata_source": checkpoint_metadata_dir,
            "key_state_config_source": key_state_config,
        }

    train_config_name = usr_args.get("train_config_name")
    model_name = usr_args.get("model_name")
    checkpoint_id = usr_args.get("checkpoint_id")
    if not train_config_name or not model_name or checkpoint_id is None:
        return {}

    checkpoint_dir = (
        Path("policy/pi05/checkpoints")
        / str(train_config_name)
        / str(model_name)
        / str(checkpoint_id)
    )
    checkpoint_run_dir = checkpoint_dir.parent
    checkpoint_metadata_dir = checkpoint_run_dir / "metadata"
    key_state_config = checkpoint_metadata_dir / "rmbench_data_meta" / "key_state_config.yaml"
    return {
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_run_dir": checkpoint_run_dir,
        "checkpoint_metadata_source": checkpoint_metadata_dir,
        "key_state_config_source": key_state_config,
    }


def copy_checkpoint_metadata(save_dir, usr_args):
    paths = resolve_checkpoint_paths(usr_args)
    if not paths:
        return {}

    requires_key_state = requires_key_state_metadata(usr_args)
    metadata_source = paths["checkpoint_metadata_source"]
    key_state_config = paths["key_state_config_source"]

    info = {
        "checkpoint_dir": paths["checkpoint_dir"],
        "checkpoint_run_dir": paths["checkpoint_run_dir"],
        "checkpoint_metadata_source": metadata_source,
        "checkpoint_metadata_dir": None,
        "copied": False,
    }

    if not metadata_source.exists():
        if requires_key_state:
            raise FileNotFoundError(f"Missing checkpoint metadata for key-state checkpoint: {metadata_source}")
        return info

    if requires_key_state and not key_state_config.exists():
        raise FileNotFoundError(f"Missing key-state config for key-state checkpoint: {key_state_config}")

    target = save_dir / "checkpoint_metadata"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(metadata_source, target)

    info["checkpoint_metadata_dir"] = target
    info["copied"] = True
    return info


def load_yaml_file(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_key_state_config_for_wandb(path):
    payload = load_yaml_file(path)
    if isinstance(payload, dict) and "config" in payload:
        return payload["config"]
    return payload


def infer_wandb_group(save_dir):
    parts = Path(save_dir).parts
    if len(parts) >= 2 and parts[0] == "eval_result":
        return parts[1]
    return None


def default_wandb_name(usr_args, current_time):
    timestamp = current_time.replace(" ", "_").replace(":", "-")
    task = usr_args.get("task_name", "unknown_task")
    model = usr_args.get("model_name") or usr_args.get("ckpt_setting") or usr_args.get("policy_name", "policy")
    return f"eval_{task}_{model}_{timestamp}"


def build_wandb_config(save_dir, usr_args, task_args):
    metadata_dir = Path(save_dir) / "checkpoint_metadata"
    rmbench_meta_dir = metadata_dir / "rmbench_data_meta"
    return to_yaml_safe({
        "eval": {
            "task_name": usr_args.get("task_name"),
            "task_config": usr_args.get("task_config"),
            "policy_name": usr_args.get("policy_name"),
            "model_name": usr_args.get("model_name"),
            "train_config_name": usr_args.get("train_config_name"),
            "checkpoint_id": usr_args.get("checkpoint_id"),
            "ckpt_setting": usr_args.get("ckpt_setting"),
            "instruction_type": usr_args.get("instruction_type"),
            "seed": usr_args.get("seed"),
            "test_num": usr_args.get("test_num", 100),
            "eval_video_log": usr_args.get("eval_video_log", True),
            "eval_video_count": usr_args.get("eval_video_count", 5),
            "eval_video_key_state_overlay": usr_args.get("eval_video_key_state_overlay", False),
            "key_state_rollout_mode": usr_args.get("key_state_rollout_mode", "predicted"),
            "state_token_rollout_mode": usr_args.get("state_token_rollout_mode", "predicted"),
            "eval_output_dir": str(save_dir),
        },
        "runtime": usr_args.get("_runtime", {}),
        "task_args": task_args,
        "checkpoint_metadata": {
            "train_config": load_yaml_file(metadata_dir / "train_config.yaml"),
            "key_state_config": load_key_state_config_for_wandb(rmbench_meta_dir / "key_state_config.yaml"),
            "source_data_config": load_yaml_file(rmbench_meta_dir / "source_data_config.yaml"),
        },
    })


def init_eval_wandb(save_dir, current_time, usr_args, task_args):
    if not coerce_bool(usr_args.get("wandb_enabled", True)):
        return None

    import wandb

    project = (
        usr_args.get("wandb_project")
        or os.environ.get("WANDB_PROJECT")
        or "RMBench"
    )
    group = (
        usr_args.get("wandb_group")
        or os.environ.get("WANDB_RUN_GROUP")
        or infer_wandb_group(save_dir)
    )
    name = usr_args.get("wandb_name") or default_wandb_name(usr_args, current_time)
    job_type = usr_args.get("wandb_job_type", "eval")

    run = wandb.init(
        project=project,
        group=group,
        name=name,
        job_type=job_type,
        config=build_wandb_config(save_dir, usr_args, task_args),
        dir=str(save_dir),
    )
    if wandb.run is not None:
        wandb_id_path = Path(save_dir) / "wandb_id.txt"
        wandb_id_path.write_text(wandb.run.id, encoding="utf-8")
        wandb_run_dir = Path(wandb.run.dir).parent if wandb.run.dir else None
        usr_args.setdefault("_runtime", {})["wandb"] = {
            "project": project,
            "group": group,
            "name": name,
            "job_type": job_type,
            "id": wandb.run.id,
            "url": wandb.run.url,
            "local_dir": wandb_run_dir,
        }
        wandb.config.update(
            {"runtime": to_yaml_safe(usr_args.get("_runtime", {}))},
            allow_val_change=True,
        )
    return run


def iter_eval_files_for_wandb(save_dir):
    save_dir = Path(save_dir)
    root_files = [
        "_result.txt",
        "eval_log.txt",
        "episode_diagnostics.jsonl",
        "diagnostics_summary.json",
        "stdout.log",
        "config.yaml",
        "command.txt",
        "wandb_id.txt",
    ]
    for name in root_files:
        path = save_dir / name
        if path.exists():
            yield path

    metadata_dir = save_dir / "checkpoint_metadata"
    if metadata_dir.exists():
        for path in sorted(metadata_dir.rglob("*")):
            relative_path = path.relative_to(save_dir).as_posix()
            if relative_path == "checkpoint_metadata/rmbench_data_meta/key_state_config.yaml":
                continue
            if path.is_file():
                yield path


def upload_eval_to_wandb(save_dir, metrics):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    wandb.log(metrics)
    wandb.run.summary.update(metrics)

    save_dir = Path(save_dir)
    save_base_path = save_dir.parent
    for path in iter_eval_files_for_wandb(save_dir):
        try:
            wandb.save(str(path), base_path=str(save_base_path), policy="now")
        except Exception as exc:
            logging.warning("Failed to save %s to wandb: %s", path, exc)

    videos = sorted(save_dir.glob("episode*.mp4"))
    if videos:
        wandb.log({
            "eval/rollout_videos": [
                wandb.Video(str(path), fps=10, format="mp4")
                for path in videos
            ]
        })


def update_wandb_eval_run_config(save_dir):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    eval_config = load_yaml_file(Path(save_dir) / "config.yaml")
    if eval_config is not None:
        wandb.config.update(
            {"eval_run_config": to_yaml_safe(eval_config)},
            allow_val_change=True,
        )


def write_eval_run_files(save_dir, current_time, usr_args, task_args):
    runtime = dict(usr_args.get("_runtime", {}))
    runtime.update({
        "timestamp": current_time,
        "save_dir": str(save_dir),
    })
    usr_args_snapshot = {key: value for key, value in usr_args.items() if key != "_runtime"}
    snapshot = {
        "runtime": runtime,
        "usr_args": usr_args_snapshot,
        "task_args": task_args,
    }
    with (save_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_yaml_safe(snapshot), f, allow_unicode=True, sort_keys=False)

    with (save_dir / "command.txt").open("w", encoding="utf-8") as f:
        f.write(f"commit: {runtime.get('git_commit', 'unknown')}\n")
        f.write(f"cwd: {runtime.get('cwd', '')}\n")
        env = runtime.get("env", {})
        if env:
            f.write("env:\n")
            for key, value in env.items():
                f.write(f"  {key}={value}\n")
        f.write("command:\n")
        f.write(f"  {runtime.get('command', '')}\n")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_eval_video_settings(args, usr_args):
    eval_video_log = coerce_bool(usr_args.get("eval_video_log", True))
    eval_video_count = int(usr_args.get("eval_video_count", 5))
    eval_video_key_state_overlay = coerce_bool(usr_args.get("eval_video_key_state_overlay", False))
    if eval_video_count < 0:
        eval_video_count = 0

    args["eval_video_log"] = eval_video_log
    args["eval_video_count"] = eval_video_count
    args["eval_video_key_state_overlay"] = eval_video_key_state_overlay
    return eval_video_log, eval_video_count


def should_record_eval_video(args, rollout_idx):
    return (
        coerce_bool(args.get("eval_video_log", False))
        and rollout_idx < int(args.get("eval_video_count", 0))
    )


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    save_dir = resolve_eval_save_dir(usr_args, task_name, policy_name, task_config, ckpt_setting, current_time)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / "eval_log.txt"
    args["log_file"] = str(log_file)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Eval log for {task_name} | {policy_name} | {task_config} | {ckpt_setting}\n")
        f.write(f"Timestamp: {current_time}\n\n")

    eval_video_log, eval_video_count = get_eval_video_settings(args, usr_args)
    if eval_video_log and eval_video_count > 0:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir
    else:
        args.pop("eval_video_save_dir", None)

    checkpoint_metadata = copy_checkpoint_metadata(save_dir, usr_args)
    if checkpoint_metadata:
        usr_args.setdefault("_runtime", {})["checkpoint"] = checkpoint_metadata

    wandb_run = init_eval_wandb(save_dir, current_time, usr_args, args)
    write_eval_run_files(save_dir, current_time, usr_args, args)
    update_wandb_eval_run_config(save_dir)

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    diagnostics_recorder = EvalDiagnosticsRecorder(save_dir)

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_args.get("test_num", 100)
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num, task_total_reward = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   diagnostics_recorder=diagnostics_recorder)
    suc_nums.append(suc_num)
    diagnostics_summary = diagnostics_recorder.write_summary()

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")

        # Success Rate
        success_rates = (np.asarray(suc_nums, dtype=float) / float(test_num)).reshape(-1)
        for sr in success_rates:
            file.write(f"Success Rate: {sr}\n")

        file.write("\n")  # 分隔一下

        # Reward
        rewards = task_total_reward / test_num
        if np.isscalar(rewards):
            file.write(f"Reward: {float(rewards)}\n")
        else:
            rewards = np.asarray(rewards, dtype=float).reshape(-1)
            for r in rewards:
                file.write(f"Reward: {r}\n")

        file.write("\n")
        file.write(EvalDiagnosticsRecorder.format_summary(diagnostics_summary))

    print(f"Data has been saved to {file_path}")
    metrics = {
        "eval/success_rate": float(success_rates[0]),
        "eval/success_count": int(suc_num),
        "eval/test_num": int(test_num),
        "eval/reward": float(rewards if np.isscalar(rewards) else np.asarray(rewards).reshape(-1)[0]),
    }
    metrics.update(EvalDiagnosticsRecorder.wandb_metrics(diagnostics_summary))
    upload_eval_to_wandb(save_dir, metrics)
    if wandb_run is not None:
        import wandb
        wandb.finish()
    # return task_reward

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                diagnostics_recorder=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", e)
                print(stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        record_video = should_record_eval_video(args, TASK_ENV.test_num)
        if record_video and TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        task_total_reward += TASK_ENV.max_reward
        if record_video:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m", " | max reward:", TASK_ENV.max_reward)
            result_str = "Success"
        else:
            print("\033[91mFail!\033[0m", " | max reward:", TASK_ENV.max_reward)
            result_str = "Fail"

        diagnostic_record = None
        if diagnostics_recorder is not None:
            diagnostic_record = diagnostics_recorder.record_episode(
                TASK_ENV,
                episode_id=now_id,
                seed=now_seed,
                success=succ,
            )
            failure_reason = diagnostic_record["diagnostics"]["primary_failure_reason"]
            print(f"Diagnostic: {failure_reason}")

        log_file = args.get("log_file", None)
        if log_file is not None:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    diagnostic_suffix = ""
                    if diagnostic_record is not None:
                        diagnostic_suffix = (
                            ", failure_reason="
                            + str(diagnostic_record["diagnostics"]["primary_failure_reason"])
                        )
                    f.write(
                        f"episode_id={now_id}, "
                        f"seed={now_seed}, "
                        f"result={result_str}{diagnostic_suffix}\n"
                    )
            except Exception as e:
                print(f"[Log Warning] Failed to write log: {e}")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, task_total_reward


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError:
                pass
            override_dict[key] = value
        return override_dict

    overrides = {}
    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    config["_runtime"] = {
        "config_path": args.config,
        "overrides": overrides,
        "command": " ".join(shlex.quote(item) for item in sys.argv),
        "cwd": os.getcwd(),
        "git_commit": get_git_commit(),
        "env": get_runtime_env(),
    }

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
