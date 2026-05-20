#!/bin/bash
echo "=== $(date) ==="
echo ""
echo "GPU Utilization:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | column -t -s,
echo ""
echo "Per-task success rates:"
for task in observe_and_pickup rearrange_blocks put_back_block swap_blocks swap_T battery_try blocks_ranking_try cover_blocks press_button; do
    log="/tmp/eval_${task}.log"
    last=$(grep "Success rate:" "$log" 2>/dev/null | tail -1)
    if [ -z "$last" ]; then
        echo "  $task: (still initializing)"
    else
        echo "  $task: $last"
    fi
done
