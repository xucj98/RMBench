#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DM05_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MODEL_PATH="${DM05_MODEL_PATH:-/mnt/public3/xcj/rmbench/dm05/checkpoints/DM05-robotwin2}"
OUTPUT_DIR="${DM05_OUTPUT_DIR:-/mnt/public3/xcj/rmbench/dm05/user_checkpoints/swap_blocks}"
TRAIN_STEPS="${DM05_TRAIN_STEPS:-10000}"
SAVE_STEPS="${DM05_SAVE_STEPS:-2500}"
BATCH_SIZE="${DM05_BATCH_SIZE:-2}"
NORM_STATS_ROOT="${DM05_NORM_STATS_ROOT:-/mnt/public3/xcj/rmbench/dm05/norm_stats}"
LOGGING_STEPS="${DM05_LOGGING_STEPS:-100}"

cd "${DM05_ROOT}"
script/dm05_launcher.sh \
  --exp playground/dm05_rmbench.py \
  --nproc_per_node 4 \
  --task train \
  --data-config.dataset-name rmbench_swap_blocks \
  --data-config.norm-stats-root "${NORM_STATS_ROOT}" \
  --model-config.model-name-or-path "${MODEL_PATH}" \
  --model-config.chunk-size 50 \
  --trainer-config.per-device-train-batch-size "${BATCH_SIZE}" \
  --trainer-config.num-train-steps "${TRAIN_STEPS}" \
  --trainer-config.save-steps "${SAVE_STEPS}" \
  --trainer-config.logging-steps "${LOGGING_STEPS}" \
  --trainer-config.output-dir "${OUTPUT_DIR}" \
  --trainer-config.wandb-project ""
