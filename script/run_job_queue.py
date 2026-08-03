#!/usr/bin/env python3
"""Small persistent GPU queue for repository experiment manifests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _gpu_snapshot() -> dict[int, dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in output.splitlines():
        index, total, used, util = (int(value.strip()) for value in line.split(","))
        result[index] = {"total_mib": total, "used_mib": used, "free_mib": total - used, "util": util}
    return result


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _eligible(snapshot: dict[str, int], requirements: dict[str, Any]) -> bool:
    placement = requirements.get("placement", "exclusive")
    min_free = int(requirements.get("min_free_memory_mib", 70_000))
    max_util = int(requirements.get("max_existing_util", 10))
    if placement == "exclusive" and snapshot["used_mib"] > int(requirements.get("max_used_memory_mib", 1024)):
        return False
    return snapshot["free_mib"] >= min_free and snapshot["util"] <= max_util


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU indices")
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs_path = args.jobs if args.jobs.is_absolute() else ROOT / args.jobs
    state_path = args.state if args.state.is_absolute() else ROOT / args.state
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    state: dict[str, Any] = {
        "jobs_file": str(jobs_path),
        "gpus": gpu_ids,
        "started_at": _now(),
        "pending": [job["name"] for job in jobs],
        "running": {},
        "succeeded": [],
        "failed": [],
    }
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    pending = list(jobs)
    running: dict[int, tuple[subprocess.Popen, Any, dict[str, Any]]] = {}
    stable_counts = {gpu: 0 for gpu in gpu_ids}
    _write_state(state_path, state)

    while pending or running:
        for gpu, (process, log_file, record) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            record["return_code"] = return_code
            record["end_time"] = _now()
            state["running"].pop(record["name"], None)
            (state["succeeded"] if return_code == 0 else state["failed"]).append(record)
            del running[gpu]

        snapshots = _gpu_snapshot()
        for gpu in gpu_ids:
            if gpu in running or not pending:
                continue
            requirements = pending[0].get("gpu_requirements", {})
            snapshot = snapshots[gpu]
            stable_counts[gpu] = stable_counts[gpu] + 1 if _eligible(snapshot, requirements) else 0
            if stable_counts[gpu] < int(requirements.get("stable_polls", 2)):
                continue

            job = pending.pop(0)
            cwd = ROOT / job.get("cwd", ".")
            log_path = ROOT / job["log_path"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            for key, value in job.get("env", {}).items():
                env[key] = str(value).format(assigned_gpu=gpu)
            process = subprocess.Popen(
                job["cmd"], cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
            )
            pid_path = ROOT / job.get("pid_path", str(log_path.parent / "train.pid"))
            pid_path.parent.mkdir(parents=True, exist_ok=True)
            pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            record = {
                "name": job["name"],
                "pid": process.pid,
                "assigned_gpu": gpu,
                "start_time": _now(),
                "log_path": str(log_path),
                "pid_path": str(pid_path),
                "launch_snapshot": snapshot,
                "command": job["cmd"],
            }
            running[gpu] = (process, log_file, record)
            state["running"][job["name"]] = record
            state["pending"] = [item["name"] for item in pending]
            stable_counts[gpu] = 0
            print(f"started {job['name']} on GPU {gpu}: pid={process.pid}", flush=True)

        _write_state(state_path, state)
        if pending or running:
            time.sleep(args.poll_seconds)

    state["finished_at"] = _now()
    _write_state(state_path, state)
    if state["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
