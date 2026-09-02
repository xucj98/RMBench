#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DM05_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT="${1:?usage: serve_rmbench_swap_blocks.sh CHECKPOINT [PORT]}"
PORT="${2:-7891}"
NORM_STATS_ROOT="${DM05_NORM_STATS_ROOT:-/mnt/public3/xcj/rmbench/dm05/norm_stats}"
VISION_ATTN="${DM05_VISION_ATTN:-flash_attention_2}"

cd "${DM05_ROOT}"
script/dm05_launcher.sh \
  --exp playground/dm05_rmbench.py \
  --task inference \
  --data-config.dataset-name rmbench_swap_blocks \
  --data-config.norm-stats-root "${NORM_STATS_ROOT}" \
  --model-config.model-name-or-path "${CHECKPOINT}" \
  --model-config.chunk-size 50 \
  --model-config.vision-attn-implementation "${VISION_ATTN}" \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port "${PORT}"
