#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-policy/pi05/.venv/bin/python}"

"${PYTHON_BIN}" policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
  --config experiments/cover_blocks_key_state_design/converter_configs/cover_blocks_exec2_attr3_no_phase.yaml \
  --overrides \
    dataset.episodes=1 \
    dataset.repo_id=cover_blocks_demo_clean_state_key_state_exec2_attr3_no_phase_smoke

"${PYTHON_BIN}" policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
  --config experiments/cover_blocks_key_state_design/converter_configs/cover_blocks_phase_exec2_attr3.yaml \
  --overrides \
    dataset.episodes=1 \
    dataset.repo_id=cover_blocks_demo_clean_state_key_state_phase_exec2_attr3_smoke
