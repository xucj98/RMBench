# pi0_full_key_state

本批次用于评估 pi0 full finetune 在 key-state 数据上的性能。当前先跑：

```text
rearrange_blocks
```

该任务复用 `pi0_key_state_baseline` 已生成的 LeRobot key-state 数据，不重新生成
RMBench demo，也不重新转换数据。

## 实验范围

```text
policy: pi05 deploy pi0 checkpoint
train recipe: pi0 full finetune
train_config_name: pi0_full_key_state
batch_size: 32
num_train_steps: 30000
gpu: one run per GPU
xla_mem_fraction: 0.95
wandb project: RMBench
wandb group: pi0_full_key_state
eval: pending
```

## 数据和 Assets

使用已有 LeRobot repo：

```text
rearrange_blocks: repo_id=rearrange_blocks_demo_clean_state_key_state
```

正式训练前使用 `max_frames=10000` 计算 norm stats，并写入：

```text
policy/pi05/assets/pi0_full_key_state/rearrange_blocks_demo_clean_state_key_state/norm_stats.json
```

## 训练

checkpoint 路径规则：

```text
policy/pi05/checkpoints/pi0_full_key_state/<exp_name>
```

正式训练状态：

| Task | Repo ID | GPU | Status | wandb id | Checkpoint | stdout |
| --- | --- | ---: | --- | --- | --- | --- |
| `rearrange_blocks` | `rearrange_blocks_demo_clean_state_key_state` | 4 | running | `sza9j7fr` | `policy/pi05/checkpoints/pi0_full_key_state/pi0_full_key_state_rearrange_blocks` | `policy/pi05/logs/pi0_full_key_state/pi0_full_key_state_rearrange_blocks.stdout.log` |

训练命令、git commit、cwd、环境变量和 resolved train config 由 checkpoint metadata
自动记录；README 不重复手写这些字段。

## 评测

```text
pending
```
