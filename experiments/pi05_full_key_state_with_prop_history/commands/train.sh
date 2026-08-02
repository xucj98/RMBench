#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <rearrange_blocks|put_back_block> <gpu>" >&2
  exit 2
fi

task="$1"
gpu="$2"
case "$task" in
  rearrange_blocks|put_back_block) ;;
  *) echo "unsupported task: $task" >&2; exit 2 ;;
esac

config="pi05_full_key_state_with_prop_history"
repo_id="${task}_demo_clean_state_key_state"
run_dir="policy/pi05/checkpoints/${config}/${task}"
mkdir -p "$run_dir"

setsid bash -lc "cd policy/pi05 && env CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONPATH=src WANDB_PROJECT=RMBench WANDB_RUN_GROUP=${config} .venv/bin/python scripts/train.py ${config} --exp-name=${task} --data.repo-id=${repo_id} --resume" \
  > "${run_dir}/train.stdout.log" 2>&1 &
echo $! > "${run_dir}/train.pid"
echo "started ${task} on GPU ${gpu}: pid=$(cat "${run_dir}/train.pid")"
