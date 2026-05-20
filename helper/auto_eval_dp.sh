#!/bin/bash
# Auto-evaluate DP models after training completes

cd /root/projects/RMBench

# Ensure curobo CUDA extensions can find torch libraries
export LD_LIBRARY_PATH="/root/miniconda3/envs/RMBench/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"
export PYTHONWARNINGS="ignore::UserWarning"

# Use a free GPU (avoid GPU 0 which still has training residue)
export CUDA_VISIBLE_DEVICES=2

TASKS=(
    observe_and_pickup
    rearrange_blocks
    put_back_block
    swap_blocks
    swap_T
    battery_try
    blocks_ranking_try
    cover_blocks
    press_button
)

for task in "${TASKS[@]}"; do
    CKPT_DIR="policy/DP/checkpoints/${task}-demo_clean-50-0"
    if [ -f "${CKPT_DIR}/600.ckpt" ]; then
        echo "=========================================="
        echo "Evaluating $task with checkpoint 600"
        echo "=========================================="
        /root/miniconda3/envs/RMBench/bin/python script/eval_policy.py --config policy/DP/deploy_policy.yml \
            --overrides \
            --task_name ${task} \
            --task_config demo_clean \
            --ckpt_setting default \
            --expert_data_num 50 \
            --seed 0
    else
        echo "Checkpoint not ready for $task"
    fi
done
