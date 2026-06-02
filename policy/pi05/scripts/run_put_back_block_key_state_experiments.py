"""Launch the current put_back_block key-state pi0 LoRA experiment batch.

The launcher converts the configured key-state dataset variants, validates the
generated LeRobot data, computes norm stats, and starts one training job per
assigned GPU while keeping GPU0 free by default. Paths default to the workspace
storage entry and can be overridden from the command line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = WORKSPACE_ROOT / "storage/state_augmented_data/put_back_block/demo_clean_state"
DEFAULT_CKPT_BASE = WORKSPACE_ROOT / "storage/pi0_checkpoints"
DEFAULT_RUN_ROOT = WORKSPACE_ROOT / "storage/pi0_key_state_runs"


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "default",
        "config": "pi0_aloha_put_back_block_key_state_default_lora",
        "repo": "put_back_block_demo_clean_key_state_default",
        "exp": "pi0_put_back_block_key_state_default",
        "gpu": 1,
        "convert": {
            "phase_input_policy": "gt",
            "mat_input_policy": "unknown_until_wmat_end",
            "wmat_margin_frames": 0,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
        },
    },
    {
        "name": "mat_first",
        "config": "pi0_aloha_put_back_block_key_state_mat_first_lora",
        "repo": "put_back_block_demo_clean_key_state_mat_first",
        "exp": "pi0_put_back_block_key_state_mat_first",
        "gpu": 2,
        "convert": {
            "phase_input_policy": "gt",
            "mat_input_policy": "unknown_first_frame_only",
            "wmat_margin_frames": 0,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
        },
    },
    {
        "name": "mat_hash_p50",
        "config": "pi0_aloha_put_back_block_key_state_mat_hash_p50_lora",
        "repo": "put_back_block_demo_clean_key_state_mat_hash_p50",
        "exp": "pi0_put_back_block_key_state_mat_hash_p50",
        "gpu": 3,
        "convert": {
            "phase_input_policy": "gt",
            "mat_input_policy": "early_hash_mix",
            "mat_unknown_prob": 0.5,
            "wmat_margin_frames": 0,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
            "phase_boundary_jitter_seed": 0,
        },
    },
    {
        "name": "wmat_margin10",
        "config": "pi0_aloha_put_back_block_key_state_wmat_margin10_lora",
        "repo": "put_back_block_demo_clean_key_state_wmat_margin10",
        "exp": "pi0_put_back_block_key_state_wmat_margin10",
        "gpu": 4,
        "convert": {
            "phase_input_policy": "gt",
            "mat_input_policy": "unknown_until_wmat_end",
            "wmat_margin_frames": 10,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
        },
    },
    {
        "name": "wmat_margin20",
        "config": "pi0_aloha_put_back_block_key_state_wmat_margin20_lora",
        "repo": "put_back_block_demo_clean_key_state_wmat_margin20",
        "exp": "pi0_put_back_block_key_state_wmat_margin20",
        "gpu": None,
        "convert": {
            "phase_input_policy": "gt",
            "mat_input_policy": "unknown_until_wmat_end",
            "wmat_margin_frames": 20,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
        },
    },
    {
        "name": "phase_lag10",
        "config": "pi0_aloha_put_back_block_key_state_phase_lag10_lora",
        "repo": "put_back_block_demo_clean_key_state_phase_lag10",
        "exp": "pi0_put_back_block_key_state_phase_lag10",
        "gpu": 5,
        "convert": {
            "phase_input_policy": "lag_after_boundary",
            "lag_window_frames": 10,
            "mat_input_policy": "unknown_until_wmat_end",
            "wmat_margin_frames": 0,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
        },
    },
    {
        "name": "phase_lag20",
        "config": "pi0_aloha_put_back_block_key_state_phase_lag20_lora",
        "repo": "put_back_block_demo_clean_key_state_phase_lag20",
        "exp": "pi0_put_back_block_key_state_phase_lag20",
        "gpu": 6,
        "convert": {
            "phase_input_policy": "lag_after_boundary",
            "lag_window_frames": 20,
            "mat_input_policy": "unknown_until_wmat_end",
            "wmat_margin_frames": 0,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 0,
        },
    },
    {
        "name": "phase_jitter5",
        "config": "pi0_aloha_put_back_block_key_state_phase_jitter5_lora",
        "repo": "put_back_block_demo_clean_key_state_phase_jitter5",
        "exp": "pi0_put_back_block_key_state_phase_jitter5",
        "gpu": 7,
        "convert": {
            "phase_input_policy": "gt",
            "mat_input_policy": "unknown_until_wmat_end",
            "wmat_margin_frames": 0,
            "key_output_mode": "per_step",
            "phase_boundary_jitter_frames": 5,
            "phase_boundary_jitter_seed": 0,
        },
    },
]


CONVERT_CODE = r"""
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location(
    "ks_converter", "examples/aloha_real/convert_robotwin_key_state_to_lerobot.py"
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

kwargs = json.loads(sys.argv[1])
kwargs["source_dir"] = Path(kwargs["source_dir"])
kwargs["dataset_config"] = mod.DatasetConfig(**kwargs.pop("dataset_config"))
mod.convert(mod.ConvertConfig(**kwargs))
"""


def _base_env(disable_gpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env.pop("WANDB_MODE", None)
    env.pop("WANDB_DISABLED", None)
    if disable_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
    return env


def _run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
    elapsed = time.time() - start
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {cmd}; log={log_path}")
    print(f"[done] {log_path.name} elapsed={elapsed / 60:.1f} min", flush=True)


def _convert_variant(
    variant: dict[str, Any],
    *,
    source_dir: Path,
    log_dir: Path,
    dataset_config: dict[str, Any],
) -> None:
    kwargs = {
        "source_dir": str(source_dir),
        "repo_id": variant["repo"],
        "episodes": 50,
        "instruction_type": "seen",
        "mode": "image",
        "dataset_config": dataset_config,
    }
    kwargs.update(variant["convert"])
    _run_logged(
        [sys.executable, "-c", CONVERT_CODE, json.dumps(kwargs)],
        log_dir / f"convert_{variant['name']}.log",
        _base_env(disable_gpu=True),
    )


def _phase_after_action(frame_idx: int, b01: int, b12: int) -> int:
    after_action = frame_idx + 1
    if after_action < b01:
        return 0
    if after_action < b12:
        return 1
    return 2


def _validate_variant(variant: dict[str, Any]) -> dict[str, Any]:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    repo_path = Path(HF_LEROBOT_HOME) / variant["repo"]
    meta_path = repo_path / "meta" / "key_state_config.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta["episodes"]) != 50:
        raise AssertionError(f"{variant['name']}: expected 50 episodes, got {len(meta['episodes'])}")

    total_rows = 0
    mat_counts: dict[int, int] = {}
    for episode in meta["episodes"]:
        idx = int(episode["episode_idx"])
        parquet_path = repo_path / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        if table.num_rows != int(episode["frames"]):
            raise AssertionError(f"{variant['name']} ep{idx}: rows {table.num_rows} != {episode['frames']}")

        states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if states.shape[1] != 32 or actions.shape[1] != 32:
            raise AssertionError(f"{variant['name']} ep{idx}: bad shapes {states.shape} {actions.shape}")

        for values, label in ((states, "state"), (actions, "action")):
            phase = values[:, 14:17]
            mat = values[:, 17:22]
            pad = values[:, 22:32]
            if not np.allclose(phase.sum(axis=1), 1.0):
                raise AssertionError(f"{variant['name']} ep{idx}: {label} phase not one-hot")
            if not np.allclose(mat.sum(axis=1), 1.0):
                raise AssertionError(f"{variant['name']} ep{idx}: {label} mat not one-hot")
            if not np.allclose(pad, 0.0):
                raise AssertionError(f"{variant['name']} ep{idx}: {label} pad nonzero")

        b01 = int(episode["b01"])
        b12 = int(episode["b12"])
        expected_phase = np.asarray([_phase_after_action(f, b01, b12) for f in range(table.num_rows)])
        got_phase = actions[:, 14:17].argmax(axis=1)
        if not np.array_equal(got_phase, expected_phase):
            raise AssertionError(f"{variant['name']} ep{idx}: action phase target mismatch")

        got_mat = actions[:, 17:22].argmax(axis=1)
        mat_id = int(episode["mat_id"])
        if not np.all(got_mat == mat_id):
            raise AssertionError(f"{variant['name']} ep{idx}: action mat target mismatch")

        mat_counts[mat_id] = mat_counts.get(mat_id, 0) + 1
        total_rows += table.num_rows

    return {
        "repo": variant["repo"],
        "episodes": 50,
        "frames": total_rows,
        "mat_episode_counts": mat_counts,
    }


def _compute_norm_stats(variant: dict[str, Any], *, log_dir: Path) -> None:
    _run_logged(
        [sys.executable, "scripts/compute_norm_stats.py", "--config-name", variant["config"]],
        log_dir / f"norm_{variant['name']}.log",
        _base_env(disable_gpu=True),
    )


def _train_cmd(variant: dict[str, Any], ckpt_base: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/train.py",
        variant["config"],
        f"--exp-name={variant['exp']}",
        f"--checkpoint-base-dir={ckpt_base}",
        "--overwrite",
    ]


def _launch_training(variant: dict[str, Any], *, gpu: int, log_dir: Path, run_dir: Path, ckpt_base: Path) -> dict[str, Any]:
    env = _base_env(disable_gpu=False)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
    log_path = log_dir / f"train_{variant['name']}_gpu{gpu}.log"
    log = log_path.open("w", encoding="utf-8")
    cmd = _train_cmd(variant, ckpt_base)
    log.write(
        "$ CUDA_VISIBLE_DEVICES="
        + str(gpu)
        + " XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONPATH=src "
        + " ".join(cmd)
        + "\n"
    )
    log.flush()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    log.close()
    (run_dir / f"{variant['name']}.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(f"[train] launched {variant['name']} gpu={gpu} pid={proc.pid} log={log_path}", flush=True)
    return {
        "name": variant["name"],
        "config": variant["config"],
        "repo": variant["repo"],
        "exp": variant["exp"],
        "gpu": gpu,
        "pid": proc.pid,
        "log": str(log_path),
        "checkpoint_dir": str(ckpt_base / variant["config"] / variant["exp"]),
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")


def run_queue_monitor(state_path: Path) -> None:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run_dir = Path(state["run_dir"])
    log_dir = Path(state["log_dir"])
    ckpt_base = Path(state["checkpoint_base_dir"])
    monitor_log = Path(state["monitor_log"])
    _append_log(monitor_log, "queue monitor started")
    while state["queued"]:
        for item in list(state["running"]):
            if not _pid_alive(int(item["pid"])):
                gpu = int(item["gpu"])
                state["running"].remove(item)
                variant = state["queued"].pop(0)
                launched = _launch_training(variant, gpu=gpu, log_dir=log_dir, run_dir=run_dir, ckpt_base=ckpt_base)
                state.setdefault("launched_from_queue", []).append(launched)
                state["running"].append({"name": launched["name"], "gpu": launched["gpu"], "pid": launched["pid"]})
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                break
        time.sleep(300)
    _append_log(monitor_log, "queue monitor finished launching queued jobs")


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_root / time.strftime("%Y%m%d_%H%M%S")
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    variants = json.loads(json.dumps(VARIANTS))
    if args.use_gpu0_for_queue:
        for variant in variants:
            if variant["name"] == "wmat_margin20":
                variant["gpu"] = 0

    dataset_config = {
        "use_videos": True,
        "tolerance_s": 0.0001,
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
        "video_backend": None,
    }
    manifest = {
        "run_dir": str(run_dir),
        "source_dir": str(args.source_dir),
        "checkpoint_base_dir": str(args.checkpoint_base_dir),
        "gpu0_reserved": not args.use_gpu0_for_queue,
        "dataset_config": dataset_config,
        "variants": variants,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[driver] run_dir={run_dir}", flush=True)
    print(f"[driver] variants={len(variants)}", flush=True)

    if not args.source_dir.exists():
        raise FileNotFoundError(args.source_dir)
    for idx in range(50):
        episode_path = args.source_dir / "data" / f"episode{idx}.hdf5"
        if not episode_path.exists():
            raise FileNotFoundError(episode_path)

    if args.skip_convert:
        print("[driver] skipping dataset conversion", flush=True)
    else:
        print("[driver] converting datasets...", flush=True)
        with ThreadPoolExecutor(max_workers=args.conversion_workers) as pool:
            futures = {
                pool.submit(
                    _convert_variant,
                    variant,
                    source_dir=args.source_dir,
                    log_dir=log_dir,
                    dataset_config=dataset_config,
                ): variant
                for variant in variants
            }
            for future in as_completed(futures):
                variant = futures[future]
                try:
                    future.result()
                except Exception:
                    print(
                        f"[driver] conversion failed for {variant['name']}; "
                        f"see {log_dir / ('convert_' + variant['name'] + '.log')}",
                        flush=True,
                    )
                    raise
        print("[driver] all conversions finished", flush=True)

    print("[driver] validating datasets...", flush=True)
    validation = {}
    for variant in variants:
        validation[variant["name"]] = _validate_variant(variant)
        print(
            f"[valid] {variant['name']}: frames={validation[variant['name']]['frames']} "
            f"mats={validation[variant['name']]['mat_episode_counts']}",
            flush=True,
        )
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    if args.skip_norm:
        print("[driver] skipping norm stats", flush=True)
    else:
        print("[driver] computing norm stats...", flush=True)
        with ThreadPoolExecutor(max_workers=args.norm_workers) as pool:
            futures = {pool.submit(_compute_norm_stats, variant, log_dir=log_dir): variant for variant in variants}
            for future in as_completed(futures):
                variant = futures[future]
                try:
                    future.result()
                except Exception:
                    print(
                        f"[driver] norm stats failed for {variant['name']}; "
                        f"see {log_dir / ('norm_' + variant['name'] + '.log')}",
                        flush=True,
                    )
                    raise
        print("[driver] norm stats finished", flush=True)

    running = []
    queued = []
    for variant in variants:
        if variant["gpu"] is None:
            queued.append(variant)
        else:
            running.append(
                _launch_training(
                    variant,
                    gpu=int(variant["gpu"]),
                    log_dir=log_dir,
                    run_dir=run_dir,
                    ckpt_base=args.checkpoint_base_dir,
                )
            )

    training_manifest = {
        "running": running,
        "queued": queued,
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
        "checkpoint_base_dir": str(args.checkpoint_base_dir),
    }
    (run_dir / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2), encoding="utf-8")

    if queued:
        queue_state = {
            "run_dir": str(run_dir),
            "log_dir": str(log_dir),
            "checkpoint_base_dir": str(args.checkpoint_base_dir),
            "running": [{"name": item["name"], "gpu": item["gpu"], "pid": item["pid"]} for item in running],
            "queued": queued,
            "monitor_log": str(log_dir / "queue_monitor.log"),
        }
        queue_state_path = run_dir / "queue_state.json"
        queue_state_path.write_text(json.dumps(queue_state, indent=2), encoding="utf-8")
        monitor_stdout = (log_dir / "queue_monitor.stdout.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--queue-monitor-state", str(queue_state_path)],
            cwd=ROOT,
            env=_base_env(disable_gpu=False),
            stdout=monitor_stdout,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        monitor_stdout.close()
        (run_dir / "queue_monitor.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        print(f"[queue] monitor pid={proc.pid}; queued={[v['name'] for v in queued]}", flush=True)

    print("[driver] done: trainings launched/queued", flush=True)
    print(f"[driver] manifest={run_dir / 'training_manifest.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--checkpoint-base-dir", type=Path, default=DEFAULT_CKPT_BASE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--conversion-workers", type=int, default=4)
    parser.add_argument("--norm-workers", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--use-gpu0-for-queue", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-norm", action="store_true")
    parser.add_argument("--queue-monitor-state", type=Path)
    args = parser.parse_args()
    if args.queue_monitor_state:
        run_queue_monitor(args.queue_monitor_state)
    else:
        run(args)


if __name__ == "__main__":
    main()
