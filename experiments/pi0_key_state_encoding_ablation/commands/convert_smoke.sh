#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-policy/pi05/.venv/bin/python}"

"${PYTHON_BIN}" policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
  --config experiments/pi0_key_state_encoding_ablation/converter_configs/cover_blocks_label_id.yaml \
  --overrides \
    dataset.episodes=1 \
    dataset.repo_id=cover_blocks_demo_clean_state_key_state_label_id_smoke

"${PYTHON_BIN}" policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
  --config experiments/pi0_key_state_encoding_ablation/converter_configs/battery_try_micro_stage_label_id.yaml \
  --overrides \
    dataset.episodes=1 \
    dataset.repo_id=battery_try_demo_clean_state_key_state_micro_stage_label_id_smoke
