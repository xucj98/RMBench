# RMBench Pi0 LoRA Baseline

Batch ID: `pi0_lora_baseline`

本批实验记录 pi0 LoRA 在 RMBench 9 个任务上的 baseline。9 个任务的 LoRA checkpoint 已经训练完成；当前正在统一复跑 100-rollout eval，每个任务前 5 个 episode 录制视频。

## 实验范围

```text
policy: pi05 deploy pi0 checkpoint
train recipe: pi0 LoRA
tasks:
  swap_blocks
  swap_T
  put_back_block
  observe_and_pickup
  rearrange_blocks
  cover_blocks
  battery_try
  press_button
  blocks_ranking_try
task_config: demo_clean_eval
instruction_type: unseen
checkpoint_id: 30000
eval: 100 rollouts, first 5 videos
```

不纳入正式 baseline：

```text
fastobs_video5 评测
video-count smoke
single-seed video test
key-state 消融
pi05 full finetune swap_blocks
```

## Checkpoint 和 Assets

本批 checkpoint 统一整理到：

```text
policy/pi05/checkpoints/pi0_lora_baseline/<task_name>/
```

每个任务使用 step `30000` 评测；目录中也保留历史 step `20000`。集中 norm stats 放在：

```text
policy/pi05/assets/pi0_lora_baseline/<repo_id>/norm_stats.json
```

checkpoint step 目录内部也保留训练时写入的 assets 副本：

```text
policy/pi05/checkpoints/pi0_lora_baseline/<task_name>/30000/assets/<repo_id>/norm_stats.json
```

为了让 eval 直接按整理后的 checkpoint 路径加载，本批新增聚合配置：

```text
train_config_name: pi0_lora_baseline
model_name: <task_name>
```

该配置只用于统一加载和后续复跑入口；模型结构、LoRA 设置和 robotwin aloha transform 与原 9 个 `pi0_aloha_<task>_lora` 配置保持一致。eval 时 norm stats 从 checkpoint step 内部的 assets 加载。

| Task | Repo ID | Checkpoint | wandb id |
| --- | --- | --- | --- |
| `swap_blocks` | `swap_blocks_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/swap_blocks/30000` | `rd0z38mb` |
| `swap_T` | `swap_T_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/swap_T/30000` | `ltl1wiho` |
| `put_back_block` | `put_back_block_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/put_back_block/30000` | `gdambjg2` |
| `observe_and_pickup` | `observe_and_pickup_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/observe_and_pickup/30000` | `1r6rl3qb` |
| `rearrange_blocks` | `rearrange_blocks_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/rearrange_blocks/30000` | `1hq968wc` |
| `cover_blocks` | `cover_blocks_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/cover_blocks/30000` | `ouvfun15` |
| `battery_try` | `battery_try_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/battery_try/30000` | `ey3f89cy` |
| `press_button` | `press_button_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/press_button/30000` | `qzefa4zv` |
| `blocks_ranking_try` | `blocks_ranking_try_demo_clean` | `policy/pi05/checkpoints/pi0_lora_baseline/blocks_ranking_try/30000` | `hdc73kyc` |

## 当前 100-Rollout Eval

启动命令：

```bash
python experiments/pi0_lora_baseline/run_eval_all.py \
  --run-tag 20260614_pi0_baseline_100 \
  --test-num 100 \
  --eval-video-count 5 \
  --xla-mem-fraction 0.4
```

只使用 GPU 5/6/7。根据 `task_config/_eval_step_limit.yml` 的 step limit 做手动均衡，每张卡串行跑 3 个任务：

| GPU | Tasks | Step-limit sum |
| ---: | --- | ---: |
| 5 | `blocks_ranking_try` -> `put_back_block` -> `observe_and_pickup` | 4250 |
| 6 | `cover_blocks` -> `battery_try` -> `swap_T` | 3100 |
| 7 | `press_button` -> `swap_blocks` -> `rearrange_blocks` | 3200 |

本轮 eval manifest：

```text
eval_result/pi0_lora_baseline/_workers_20260614_pi0_baseline_100.json
```

worker 状态：

```text
eval_result/pi0_lora_baseline/_worker_gpu5_20260614_pi0_baseline_100.json
eval_result/pi0_lora_baseline/_worker_gpu6_20260614_pi0_baseline_100.json
eval_result/pi0_lora_baseline/_worker_gpu7_20260614_pi0_baseline_100.json
```

eval result 目录规则：

```text
eval_result/pi0_lora_baseline/<task>_raw_100_video5_20260614_pi0_baseline_100/
```

本轮结果完成后，将把 `_result.txt` 中的 success rate 汇总到下表：

| Task | Result | Success | Eval result |
| --- | ---: | ---: | --- |
| `swap_blocks` | pending | pending | `eval_result/pi0_lora_baseline/swap_blocks_raw_100_video5_20260614_pi0_baseline_100` |
| `swap_T` | pending | pending | `eval_result/pi0_lora_baseline/swap_T_raw_100_video5_20260614_pi0_baseline_100` |
| `put_back_block` | pending | pending | `eval_result/pi0_lora_baseline/put_back_block_raw_100_video5_20260614_pi0_baseline_100` |
| `observe_and_pickup` | pending | pending | `eval_result/pi0_lora_baseline/observe_and_pickup_raw_100_video5_20260614_pi0_baseline_100` |
| `rearrange_blocks` | pending | pending | `eval_result/pi0_lora_baseline/rearrange_blocks_raw_100_video5_20260614_pi0_baseline_100` |
| `cover_blocks` | pending | pending | `eval_result/pi0_lora_baseline/cover_blocks_raw_100_video5_20260614_pi0_baseline_100` |
| `battery_try` | pending | pending | `eval_result/pi0_lora_baseline/battery_try_raw_100_video5_20260614_pi0_baseline_100` |
| `press_button` | pending | pending | `eval_result/pi0_lora_baseline/press_button_raw_100_video5_20260614_pi0_baseline_100` |
| `blocks_ranking_try` | pending | pending | `eval_result/pi0_lora_baseline/blocks_ranking_try_raw_100_video5_20260614_pi0_baseline_100` |

## 历史 3 任务结果

早期只验证了 3 个任务，使用旧路径和 50 rollouts。它们不是本批最终 9-task baseline，只保留作为历史记录。

| Task | Result | Success | Paper Pi0.5 | Source |
| --- | ---: | ---: | ---: | --- |
| `swap_blocks` | 7/50 | 14% | 24% | `eval_result/swap_blocks/pi05/demo_clean_eval/pi0_swap_blocks/2026-06-01 16:27:39/_result.txt` |
| `swap_T` | 8/50 | 16% | 15% | `eval_result/swap_T/pi05/demo_clean_eval/pi0_swap_T/2026-06-01 16:27:39/_result.txt` |
| `put_back_block` | 4/50 | 8% | 11% | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block/2026-06-01 16:27:39/_result.txt` |

## Fast Obs 正确性验证

fast obs 使用同一批旧 checkpoint，只改变 eval 中 action chunk 内的 observation 更新方式。该验证不混入 baseline 主表。

| Task | Original obs | Fast obs | Delta | Source |
| --- | ---: | ---: | ---: | --- |
| `swap_blocks` | 7/50 = 14% | 9/50 = 18% | +4 pp | `eval_result/swap_blocks/pi05/demo_clean_eval/pi0_swap_blocks_fastobs/2026-06-01 18:25:41/_result.txt` |
| `swap_T` | 8/50 = 16% | 8/50 = 16% | 0 pp | `eval_result/swap_T/pi05/demo_clean_eval/pi0_swap_T_fastobs/2026-06-01 18:16:10/_result.txt` |
| `put_back_block` | 4/50 = 8% | 4/50 = 8% | 0 pp | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_fastobs/2026-06-01 18:25:41/_result.txt` |

## wandb

```text
project: RMBench
group: pi0_lora_baseline
```

历史 3 个 run 的 group 不一定完整规范；剩余 6 个训练通过 `run_missing_tasks.py` 设置了 `WANDB_RUN_GROUP=pi0_lora_baseline`。
