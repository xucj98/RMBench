# RMBench Pi0 LoRA Baseline

Batch ID: `pi0_lora_baseline`

本批实验记录 pi0 LoRA 在 RMBench 三个任务上的 baseline 结果，并用同一批 checkpoint 验证 `get_obs_fast` 评测加速改动是否保持结果一致。

这里不单独创建 fast obs batch：fast obs 不是新的训练 recipe，也不是新的 baseline 模型，而是同一批 pi0 LoRA checkpoint 上的 eval implementation validation。主 baseline 结果只采用 original obs；fast obs 结果单独列为验证表，不混入 baseline 对比。后续如果要系统测试 fast obs 对更多 policy、task 或 seed 的影响，应另建单独 batch。

## 实验范围

正式 baseline 范围：

```text
policy: pi05 deploy pi0 checkpoint
train recipe: pi0 LoRA
tasks: swap_blocks, swap_T, put_back_block
task_config: demo_clean_eval
instruction_type: unseen
checkpoint_id: 30000
test_num: 50
```

不纳入正式 baseline：

```text
fastobs_video5 评测
video-count smoke
single-seed video test
key-state 消融
pi05 full finetune swap_blocks
```

## 检查点

| Task | Train config | Checkpoint | wandb id |
| --- | --- | --- | --- |
| `swap_blocks` | `pi0_aloha_swap_blocks_lora` | `policy/pi05/checkpoints/pi0_aloha_swap_blocks_lora/pi0_swap_blocks/30000` | `rd0z38mb` |
| `swap_T` | `pi0_aloha_swap_T_lora` | `policy/pi05/checkpoints/pi0_aloha_swap_T_lora/pi0_swap_T/30000` | `ltl1wiho` |
| `put_back_block` | `pi0_aloha_put_back_block_lora` | `policy/pi05/checkpoints/pi0_aloha_put_back_block_lora/pi0_put_back_block/30000` | `gdambjg2` |

这些 checkpoint 目录下也保留了 step `20000`，但本批 eval 使用 `checkpoint_id: 30000`。

## 运行方式

训练单个任务：

```bash
cd policy/pi05
PYTHONPATH=src .venv/bin/python scripts/train.py \
  <train_config_name> \
  --exp-name=<model_name> \
  --checkpoint-base-dir=checkpoints
```

其中：

```text
<train_config_name>: pi0_aloha_swap_blocks_lora | pi0_aloha_swap_T_lora | pi0_aloha_put_back_block_lora
<model_name>: pi0_swap_blocks | pi0_swap_T | pi0_put_back_block
```

评测单个任务：

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  policy/pi05/.venv/bin/python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name <task> \
  --task_config demo_clean_eval \
  --train_config_name <train_config_name> \
  --model_name <model_name> \
  --ckpt_setting <model_name> \
  --seed 0 \
  --policy_name pi05 \
  --test_num 50
```

评测配置来自：

```text
policy/pi05/deploy_policy.yml
```

关键参数：

```text
checkpoint_id: 30000
pi0_step: 50
instruction_type: unseen
eval_video_count: 5
```

## 基线结果

主 baseline 使用 original obs 评测结果。

| Task | Result | Success | Paper Pi0.5 | Source |
| --- | ---: | ---: | ---: | --- |
| `swap_blocks` | 7/50 | 14% | 24% | `eval_result/swap_blocks/pi05/demo_clean_eval/pi0_swap_blocks/2026-06-01 16:27:39/_result.txt` |
| `swap_T` | 8/50 | 16% | 15% | `eval_result/swap_T/pi05/demo_clean_eval/pi0_swap_T/2026-06-01 16:27:39/_result.txt` |
| `put_back_block` | 4/50 | 8% | 11% | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block/2026-06-01 16:27:39/_result.txt` |

三个任务的平均成功率：

```text
pi0 LoRA baseline: 12.7%
paper Pi0.5: 16.7%
```

结论：当前 pi0 LoRA baseline 只覆盖 3 个任务，不是 RMBench Table 1 的完整 Pi0.5 复现。`swap_T` 接近论文值，`swap_blocks` 和 `put_back_block` 低于论文。

## Fast Obs 正确性验证

fast obs 使用同一批 checkpoint，只改变 eval 中 action chunk 内的 observation 更新方式：跳过重复图像渲染，复用 cached image，并更新必要的 qpos / joint action 观测。该改动目标是加速 eval，不改变策略输入语义。

| Task | Original obs | Fast obs | Delta | Source |
| --- | ---: | ---: | ---: | --- |
| `swap_blocks` | 7/50 = 14% | 9/50 = 18% | +4 pp | `eval_result/swap_blocks/pi05/demo_clean_eval/pi0_swap_blocks_fastobs/2026-06-01 18:25:41/_result.txt` |
| `swap_T` | 8/50 = 16% | 8/50 = 16% | 0 pp | `eval_result/swap_T/pi05/demo_clean_eval/pi0_swap_T_fastobs/2026-06-01 18:16:10/_result.txt` |
| `put_back_block` | 4/50 = 8% | 4/50 = 8% | 0 pp | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_fastobs/2026-06-01 18:25:41/_result.txt` |

验证结论：

```text
swap_T 和 put_back_block 成功率完全一致。
swap_blocks 差异为 2/50，需按仿真随机性或 observation 更新细节差异看待。
```

因此 fast obs 可以作为 eval 加速实现继续使用，但如果结果用于论文级对比，仍应明确记录 eval variant。

## 产物规则

训练产物：

```text
policy/pi05/checkpoints/<train_config_name>/<model_name>/
```

评测产物：

```text
eval_result/<task>/pi05/demo_clean_eval/<model_name>/<timestamp>/
eval_result/<task>/pi05/demo_clean_eval/<model_name>_fastobs/<timestamp>/
```

wandb：

```text
project: RMBench
expected group for future reruns: pi0_lora_baseline
historical run ids: rd0z38mb, ltl1wiho, gdambjg2
```

## 已知问题

历史结果缺少统一 metadata：

```text
run commit: not_recorded
eval wandb id: not_recorded
batch_id in artifacts: not_recorded
```

这批结果可以通过 checkpoint、`wandb_id.txt` 和 `eval_result` 定位，但不能视为完全规范化的可复现实验记录。后续复跑时应在 checkpoint metadata、eval_result metadata 或 wandb config 中记录 `batch_id=pi0_lora_baseline`、commit、训练命令、评测命令、checkpoint 引用和 eval_result 引用。
