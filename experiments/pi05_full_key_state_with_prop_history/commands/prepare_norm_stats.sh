#!/usr/bin/env bash
set -euo pipefail

cd policy/pi05

config="pi05_full_key_state_with_prop_history"
for repo_id in \
  rearrange_blocks_demo_clean_state_key_state \
  put_back_block_demo_clean_state_key_state
do
  PYTHONPATH=src .venv/bin/python scripts/compute_norm_stats.py \
    --config-name="$config" \
    --repo-id="$repo_id" \
    --max-frames=10000
done
