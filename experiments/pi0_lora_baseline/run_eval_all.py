"""Launch pi0 LoRA baseline evals on fixed GPU workers."""

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
EVAL_PYTHON = WORKSPACE_ROOT / "policy/pi05/.venv/bin/python"
BATCH_ID = "pi0_lora_baseline"
TRAIN_CONFIG_NAME = "pi0_lora_baseline"

ASSIGNMENTS: dict[int, list[str]] = {
    5: ["blocks_ranking_try", "put_back_block", "observe_and_pickup"],
    6: ["cover_blocks", "battery_try", "swap_T"],
    7: ["press_button", "swap_blocks", "rearrange_blocks"],
}

STEP_LIMITS: dict[str, int] = {
    "observe_and_pickup": 250,
    "put_back_block": 500,
    "swap_T": 600,
    "rearrange_blocks": 700,
    "swap_blocks": 1000,
    "battery_try": 1000,
    "cover_blocks": 1500,
    "press_button": 1500,
    "blocks_ranking_try": 3500,
}


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE_ROOT, text=True).strip()


def build_eval_command(task: str, run_dir: Path, *, test_num: int, eval_video_count: int, seed: int, run_tag: str) -> list[str]:
    run_id = f"{task}_raw_{test_num}_video{eval_video_count}_{run_tag}"
    return [
        str(EVAL_PYTHON),
        "script/eval_policy.py",
        "--config",
        "policy/pi05/deploy_policy.yml",
        "--overrides",
        "--task_name",
        task,
        "--task_config",
        "demo_clean_eval",
        "--train_config_name",
        TRAIN_CONFIG_NAME,
        "--model_name",
        task,
        "--ckpt_setting",
        run_id,
        "--seed",
        str(seed),
        "--policy_name",
        "pi05",
        "--test_num",
        str(test_num),
        "--eval_video_log",
        "true" if eval_video_count > 0 else "false",
        "--eval_video_count",
        str(eval_video_count),
        "--eval_video_key_state_overlay",
        "false",
        "--eval_output_dir",
        str(run_dir),
    ]


def run_worker(args: argparse.Namespace) -> int:
    tasks = [task for task in args.worker_tasks.split(",") if task]
    batch_dir = WORKSPACE_ROOT / "eval_result" / BATCH_ID
    worker_state_path = batch_dir / f"_worker_gpu{args.worker_gpu}_{args.run_tag}.json"
    worker_log_path = batch_dir / f"_worker_gpu{args.worker_gpu}_{args.run_tag}.log"

    state: dict[str, Any] = {
        "gpu": args.worker_gpu,
        "tasks": tasks,
        "run_tag": args.run_tag,
        "test_num": args.test_num,
        "eval_video_count": args.eval_video_count,
        "completed": [],
        "failed": [],
    }
    worker_state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    with worker_log_path.open("a", encoding="utf-8") as worker_log:
        worker_log.write(f"[worker-start] gpu={args.worker_gpu} tasks={tasks}\n")
        worker_log.flush()

        for task in tasks:
            run_id = f"{task}_raw_{args.test_num}_video{args.eval_video_count}_{args.run_tag}"
            run_dir = batch_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            cmd = build_eval_command(
                task,
                run_dir,
                test_num=args.test_num,
                eval_video_count=args.eval_video_count,
                seed=args.seed,
                run_tag=args.run_tag,
            )

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
            env["PYTHONUNBUFFERED"] = "1"

            rendered = (
                f"CUDA_VISIBLE_DEVICES={args.worker_gpu} "
                f"XLA_PYTHON_CLIENT_MEM_FRACTION={args.xla_mem_fraction} "
                + " ".join(cmd)
            )
            stdout_path = run_dir / "stdout.log"
            worker_log.write(f"[task-start] task={task} run_dir={run_dir.relative_to(WORKSPACE_ROOT)}\n")
            worker_log.flush()

            start = time.time()
            with stdout_path.open("w", encoding="utf-8") as stdout:
                stdout.write("$ " + rendered + "\n")
                stdout.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=WORKSPACE_ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            result = {
                "task": task,
                "run_id": run_id,
                "run_dir": str(run_dir.relative_to(WORKSPACE_ROOT)),
                "stdout": str(stdout_path.relative_to(WORKSPACE_ROOT)),
                "returncode": proc.returncode,
                "elapsed_sec": round(time.time() - start, 3),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if proc.returncode == 0:
                state["completed"].append(result)
                worker_log.write(f"[task-done] task={task} rc=0 elapsed_sec={result['elapsed_sec']}\n")
            else:
                state["failed"].append(result)
                worker_log.write(f"[task-fail] task={task} rc={proc.returncode} elapsed_sec={result['elapsed_sec']}\n")
            worker_log.flush()
            worker_state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    return 1 if state["failed"] else 0


def launch_workers(args: argparse.Namespace) -> None:
    batch_dir = WORKSPACE_ROOT / "eval_result" / BATCH_ID
    batch_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "batch_id": BATCH_ID,
        "train_config_name": TRAIN_CONFIG_NAME,
        "commit": git_commit(),
        "run_tag": args.run_tag,
        "test_num": args.test_num,
        "eval_video_count": args.eval_video_count,
        "assignments": {
            str(gpu): {
                "tasks": tasks,
                "step_limit_sum": sum(STEP_LIMITS[task] for task in tasks),
            }
            for gpu, tasks in ASSIGNMENTS.items()
        },
        "workers": [],
    }

    for gpu, tasks in ASSIGNMENTS.items():
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-gpu",
            str(gpu),
            "--worker-tasks",
            ",".join(tasks),
            "--run-tag",
            args.run_tag,
            "--test-num",
            str(args.test_num),
            "--eval-video-count",
            str(args.eval_video_count),
            "--seed",
            str(args.seed),
            "--xla-mem-fraction",
            str(args.xla_mem_fraction),
        ]
        log_path = batch_dir / f"_launcher_gpu{gpu}_{args.run_tag}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=WORKSPACE_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        manifest["workers"].append({
            "gpu": gpu,
            "pid": proc.pid,
            "tasks": tasks,
            "launcher_log": str(log_path.relative_to(WORKSPACE_ROOT)),
        })

    manifest_path = batch_dir / f"_workers_{args.run_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    for worker in manifest["workers"]:
        print(f"gpu={worker['gpu']} pid={worker['pid']} tasks={worker['tasks']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-gpu", type=int)
    parser.add_argument("--worker-tasks", default="")
    parser.add_argument("--run-tag", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--test-num", type=int, default=100)
    parser.add_argument("--eval-video-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--xla-mem-fraction", type=float, default=0.4)
    args = parser.parse_args()

    if args.worker:
        if args.worker_gpu is None or not args.worker_tasks:
            raise ValueError("--worker requires --worker-gpu and --worker-tasks")
        raise SystemExit(run_worker(args))

    launch_workers(args)


if __name__ == "__main__":
    main()
