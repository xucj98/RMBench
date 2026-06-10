"""Prepare and launch the remaining six pi0 LoRA baseline training jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = EXPERIMENT_DIR.parents[1]
PI05_ROOT = WORKSPACE_ROOT / "policy/pi05"
PYTHON = PI05_ROOT / ".venv/bin/python"
BATCH_ID = "pi0_lora_baseline"

TASKS: list[dict[str, Any]] = [
    {
        "task": "observe_and_pickup",
        "config": "pi0_aloha_observe_and_pickup_lora",
        "repo": "observe_and_pickup_demo_clean",
        "exp": "pi0_observe_and_pickup",
        "gpu": 1,
    },
    {
        "task": "rearrange_blocks",
        "config": "pi0_aloha_rearrange_blocks_lora",
        "repo": "rearrange_blocks_demo_clean",
        "exp": "pi0_rearrange_blocks",
        "gpu": 2,
    },
    {
        "task": "cover_blocks",
        "config": "pi0_aloha_cover_blocks_lora",
        "repo": "cover_blocks_demo_clean",
        "exp": "pi0_cover_blocks",
        "gpu": 3,
    },
    {
        "task": "battery_try",
        "config": "pi0_aloha_battery_try_lora",
        "repo": "battery_try_demo_clean",
        "exp": "pi0_battery_try",
        "gpu": 4,
    },
    {
        "task": "press_button",
        "config": "pi0_aloha_press_button_lora",
        "repo": "press_button_demo_clean",
        "exp": "pi0_press_button",
        "gpu": 5,
    },
    {
        "task": "blocks_ranking_try",
        "config": "pi0_aloha_blocks_ranking_try_lora",
        "repo": "blocks_ranking_try_demo_clean",
        "exp": "pi0_blocks_ranking_try",
        "gpu": 6,
    },
]


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE_ROOT, text=True).strip()


def _base_env(*, disable_gpu: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    if disable_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
    return env


def _run_logged(cmd: list[str], log_path: Path, *, disable_gpu: bool = False, dry_run: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "$ " + " ".join(cmd)
    print(rendered, flush=True)
    if dry_run:
        return

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(rendered + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PI05_ROOT,
            env=_base_env(disable_gpu=disable_gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; see {log_path}")
    elapsed = (time.time() - start) / 60
    print(f"[done] {log_path.name} elapsed={elapsed:.1f} min", flush=True)


def _processed_dir(task: dict[str, Any]) -> Path:
    return PI05_ROOT / "processed_data" / f"{task['task']}-demo_clean-50"


def _lerobot_dir(task: dict[str, Any]) -> Path:
    return Path.home() / ".cache/huggingface/lerobot" / task["repo"]


def _norm_stats_path(task: dict[str, Any]) -> Path:
    return PI05_ROOT / "assets" / task["config"] / task["repo"] / "norm_stats.json"


def _checkpoint_dir(task: dict[str, Any]) -> Path:
    return PI05_ROOT / "checkpoints" / task["config"] / task["exp"]


def _check_raw_data(task: dict[str, Any]) -> None:
    raw_dir = WORKSPACE_ROOT / "data" / task["task"] / "demo_clean"
    for idx in range(50):
        hdf5_path = raw_dir / "data" / f"episode{idx}.hdf5"
        instruction_path = raw_dir / "instructions" / f"episode{idx}.json"
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
        if not instruction_path.exists():
            raise FileNotFoundError(instruction_path)


def _processed_ready(task: dict[str, Any]) -> bool:
    processed = _processed_dir(task)
    return all((processed / f"episode_{idx}" / f"episode_{idx}.hdf5").exists() for idx in range(50))


def _lerobot_ready(task: dict[str, Any]) -> bool:
    repo = _lerobot_dir(task)
    info = repo / "meta/info.json"
    episodes = repo / "meta/episodes.jsonl"
    return info.exists() and episodes.exists()


def _norm_ready(task: dict[str, Any]) -> bool:
    return _norm_stats_path(task).exists()


def prepare_data(args: argparse.Namespace, run_dir: Path) -> None:
    log_dir = run_dir / "logs"
    for task in TASKS:
        _check_raw_data(task)

        if args.force_process_data or not _processed_ready(task):
            _run_logged(
                [str(PYTHON), "scripts/process_data.py", task["task"], "demo_clean", "50"],
                log_dir / f"process_{task['task']}.log",
                disable_gpu=True,
                dry_run=args.dry_run,
            )
        else:
            print(f"[skip] processed_data ready: {_processed_dir(task)}", flush=True)

        if args.force_convert or not _lerobot_ready(task):
            _run_logged(
                [
                    str(PYTHON),
                    "examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py",
                    "--raw_dir",
                    str(_processed_dir(task)),
                    "--repo_id",
                    task["repo"],
                ],
                log_dir / f"convert_{task['task']}.log",
                disable_gpu=True,
                dry_run=args.dry_run,
            )
        else:
            print(f"[skip] LeRobot dataset ready: {_lerobot_dir(task)}", flush=True)

        if args.force_norm or not _norm_ready(task):
            _run_logged(
                [str(PYTHON), "scripts/compute_norm_stats.py", "--config-name", task["config"]],
                log_dir / f"norm_{task['task']}.log",
                disable_gpu=True,
                dry_run=args.dry_run,
            )
        else:
            print(f"[skip] norm stats ready: {_norm_stats_path(task)}", flush=True)


def launch_training(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    log_dir = run_dir / "logs"
    launched = []
    for task in TASKS:
        ckpt_dir = _checkpoint_dir(task)
        if ckpt_dir.exists() and not args.overwrite_checkpoints:
            raise FileExistsError(f"{ckpt_dir} exists; pass --overwrite-checkpoints to replace it")

        cmd = [
            str(PYTHON),
            "scripts/train.py",
            task["config"],
            f"--exp-name={task['exp']}",
            "--checkpoint-base-dir=checkpoints",
        ]
        if args.overwrite_checkpoints:
            cmd.append("--overwrite")

        env = _base_env(disable_gpu=False)
        env["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
        env["WANDB_RUN_GROUP"] = BATCH_ID
        env["WANDB_TAGS"] = f"{BATCH_ID},missing6"

        log_path = log_dir / f"train_{task['task']}_gpu{task['gpu']}.log"
        rendered = (
            f"$ CUDA_VISIBLE_DEVICES={task['gpu']} "
            f"XLA_PYTHON_CLIENT_MEM_FRACTION={args.xla_mem_fraction} "
            f"WANDB_RUN_GROUP={BATCH_ID} PYTHONPATH=src "
            + " ".join(cmd)
        )
        print(rendered, flush=True)
        if args.dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as log:
            log.write(rendered + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=PI05_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

        (run_dir / f"{task['task']}.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        launched.append(
            {
                **task,
                "pid": proc.pid,
                "log": str(log_path),
                "checkpoint_dir": str(ckpt_dir),
            }
        )
        print(f"[train] {task['task']} gpu={task['gpu']} pid={proc.pid} log={log_path}", flush=True)

    return launched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-process-data", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--force-norm", action="store_true")
    parser.add_argument("--overwrite-checkpoints", action="store_true")
    parser.add_argument("--xla-mem-fraction", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = PI05_ROOT / "checkpoints" / "_launch_logs" / BATCH_ID / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "batch_id": BATCH_ID,
        "commit": _git_commit(),
        "run_dir": str(run_dir),
        "tasks": [
            {
                **task,
                "processed_dir": str(_processed_dir(task)),
                "lerobot_dir": str(_lerobot_dir(task)),
                "norm_stats": str(_norm_stats_path(task)),
                "checkpoint_dir": str(_checkpoint_dir(task)),
            }
            for task in TASKS
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[run] manifest={run_dir / 'manifest.json'}", flush=True)

    if not args.skip_prepare:
        prepare_data(args, run_dir)

    launched = [] if args.skip_train else launch_training(args, run_dir)
    (run_dir / "training_manifest.json").write_text(json.dumps(launched, indent=2), encoding="utf-8")
    print(f"[done] run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
