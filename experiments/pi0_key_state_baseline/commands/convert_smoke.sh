#!/usr/bin/env bash
set -euo pipefail

configs=(
  rearrange_blocks
  swap_blocks
  battery_try
  cover_blocks
)

PYTHON_BIN="${PYTHON_BIN:-policy/pi05/.venv/bin/python}"

for task in "${configs[@]}"; do
  "${PYTHON_BIN}" policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
    --config "experiments/pi0_key_state_baseline/converter_configs/${task}.yaml" \
    --overrides \
      dataset.episodes=1 \
      dataset.repo_id="${task}_demo_clean_state_key_state_smoke"
done
