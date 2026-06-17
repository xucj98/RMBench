# pi05_full_key_state

本批次用于评估 Pi0.5 full finetune 在 key-state 数据上的性能，先跑两个任务：

```text
swap_blocks
battery_try
```

这两个任务复用 `pi0_key_state_baseline` 已生成的 LeRobot key-state 数据，不重新生成
RMBench demo，也不重新转换数据。

## 实验范围

```text
policy: pi05 deploy pi05 checkpoint
train recipe: Pi0.5 full finetune
train_config_name: full_key_state
batch_size: 32
num_train_steps: 20000
gpu: one run per GPU
xla_mem_fraction: 0.95
wandb project: RMBench
wandb group: pi05_full_key_state
eval: pending
```

## 数据和 Assets

使用已有 LeRobot repo：

```text
swap_blocks: repo_id=swap_blocks_demo_clean_state_key_state
battery_try: repo_id=battery_try_demo_clean_state_key_state
```

norm stats 已按 `full_key_state` 训练配置计算并写入：

```text
policy/pi05/assets/full_key_state/swap_blocks_demo_clean_state_key_state/norm_stats.json
policy/pi05/assets/full_key_state/battery_try_demo_clean_state_key_state/norm_stats.json
```

## 训练

checkpoint 路径规则：

```text
policy/pi05/checkpoints/full_key_state/<exp_name>
```

正式训练状态：

| Task | Repo ID | GPU | Status | wandb id | Checkpoint | stdout |
| --- | --- | ---: | --- | --- | --- | --- |
| `swap_blocks` | `swap_blocks_demo_clean_state_key_state` | 0 | running | `65ldopff` | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_swap_blocks` | `policy/pi05/logs/full_key_state/pi05_full_key_state_swap_blocks.stdout.log` |
| `battery_try` | `battery_try_demo_clean_state_key_state` | 7 | running | `2mueiwio` | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_battery_try` | `policy/pi05/logs/full_key_state/pi05_full_key_state_battery_try.stdout.log` |

训练命令、git commit、cwd、环境变量和 resolved train config 由 checkpoint metadata
自动记录；README 不重复手写这些字段。

## 评测

```text
pending
```
