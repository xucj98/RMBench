# pi05_full_baseline

本批次用于评估 Pi0.5 全量微调在 RMBench 差距较大的两个任务上的性能，并与
`pi0_lora_baseline` 和 `pi0_full_baseline` 对照。

## 实验范围

```text
policy: pi05 deploy pi05 checkpoint
train recipe: pi05 full finetune
tasks:
  rearrange_blocks
  battery_try
batch_size: 32
gpu: one run per GPU
xla_mem_fraction: 0.95
eval: pending
```

训练配置：

```text
train_config_name: pi05_full_baseline
```

该配置通过 CLI 覆盖 `--data.repo-id=<repo_id>` 和 `--exp-name=<run_name>`
复用到不同任务。checkpoint 路径规则为：

```text
policy/pi05/checkpoints/pi05_full_baseline/<run_name>/
```

## 数据和 Assets

使用已有普通 LeRobot 数据集，不重新转换数据：

```text
~/.cache/huggingface/lerobot/rearrange_blocks_demo_clean
~/.cache/huggingface/lerobot/battery_try_demo_clean
```

Pi0.5 full 使用 `adapt_to_pi=True` 的 Aloha transform，norm stats 按当前
`pi05_full_baseline` 配置重新计算：

```text
policy/pi05/assets/pi05_full_baseline/rearrange_blocks_demo_clean/norm_stats.json
policy/pi05/assets/pi05_full_baseline/battery_try_demo_clean/norm_stats.json
```

## 训练

| Task | Repo ID | GPU | Status | wandb id | Checkpoint | stdout |
| --- | --- | ---: | --- | --- | --- | --- |
| `rearrange_blocks` | `rearrange_blocks_demo_clean` | 6 | running | `rw1nmz9r` | `policy/pi05/checkpoints/pi05_full_baseline/pi05_full_baseline_rearrange_blocks` | `logs/pi05_full_baseline/pi05_full_baseline_rearrange_blocks.stdout.log` |
| `battery_try` | `battery_try_demo_clean` | 7 | running | `ew0e0sm5` | `policy/pi05/checkpoints/pi05_full_baseline/pi05_full_baseline_battery_try` | `logs/pi05_full_baseline/pi05_full_baseline_battery_try.stdout.log` |

训练命令和复现元信息由 checkpoint metadata 自动记录；README 不重复这些字段。

## 评测

```text
pending
```
