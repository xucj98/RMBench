# pi0_full_baseline

本批次用于评估 pi0 全量微调在 RMBench 差距较大的两个任务上的性能，并与
`pi0_lora_baseline` 和论文报告的 Pi0.5 baseline 对照。

## 实验范围

```text
policy: pi05 deploy pi0 checkpoint
train recipe: pi0 full finetune
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
train_config_name: pi0_full_baseline
```

该配置通过 CLI 覆盖 `--data.repo-id=<repo_id>` 和 `--exp-name=<run_name>`
复用到不同任务。checkpoint 路径规则为：

```text
policy/pi05/checkpoints/pi0_full_baseline/<run_name>/
```

## 数据和 Assets

使用已有普通 LeRobot 数据集，不重新转换数据：

```text
~/.cache/huggingface/lerobot/rearrange_blocks_demo_clean
~/.cache/huggingface/lerobot/battery_try_demo_clean
```

pi0 full 与 `pi0_lora_baseline` 使用相同的 `_robotwin_aloha_data` transform，
因此 norm stats 复用已有 pi0 LoRA baseline 的结果，并复制到当前 train config
对应的 assets 路径：

```text
policy/pi05/assets/pi0_full_baseline/rearrange_blocks_demo_clean/norm_stats.json
policy/pi05/assets/pi0_full_baseline/battery_try_demo_clean/norm_stats.json
```

## 训练

| Task | Repo ID | GPU | Status | wandb id | Checkpoint | stdout |
| --- | --- | ---: | --- | --- | --- | --- |
| `rearrange_blocks` | `rearrange_blocks_demo_clean` | 4 | running | `5qy4voef` | `policy/pi05/checkpoints/pi0_full_baseline/pi0_full_baseline_rearrange_blocks` | `logs/pi0_full_baseline/pi0_full_baseline_rearrange_blocks.stdout.log` |
| `battery_try` | `battery_try_demo_clean` | 5 | running | `1rs4tn1y` | `policy/pi05/checkpoints/pi0_full_baseline/pi0_full_baseline_battery_try` | `logs/pi0_full_baseline/pi0_full_baseline_battery_try.stdout.log` |

训练命令和复现元信息由 checkpoint metadata 自动记录；README 不重复这些字段。

## 评测

```text
pending
```
