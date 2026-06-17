# pi0_key_state_encoding_ablation

本批次用于检查 key-state 离散变量的编码方式是否影响训练效果。当前先做两个数据处理变体：

```text
cover_blocks:
  使用原来的 coarse phase / attribute 语义，但 phase 和 attribute 全部改为 label_id 数字编码。

battery_try:
  phase 改为 micro-stage 细粒度阶段划分，并使用 label_id 数字编码。
```

## 数据来源

不重新生成 RMBench 原始 demo，直接复用已有 state-augmented 数据：

```text
data/cover_blocks/demo_clean_state
data/battery_try/demo_clean_state
```

source dataset 的生成配置和命令仍以各自目录下的 `metadata/config.yaml` 和
`metadata/command.txt` 为准。

## LeRobot 转换

converter config：

```text
experiments/pi0_key_state_encoding_ablation/converter_configs/cover_blocks_label_id.yaml
experiments/pi0_key_state_encoding_ablation/converter_configs/battery_try_micro_stage_label_id.yaml
```

目标 repo：

```text
cover_blocks: repo_id=cover_blocks_demo_clean_state_key_state_label_id
battery_try:  repo_id=battery_try_demo_clean_state_key_state_micro_stage_label_id
```

当前状态：

```text
smoke conversion: passed
formal conversion: passed
```

smoke conversion 验收：

```text
cover_blocks_label_id_smoke:
  episodes=1, frames=1008, state/action=32
  phase label_id values=0..5
  red/green/blue attribute label_id values valid, padding zero

battery_try_micro_stage_label_id_smoke:
  episodes=1, frames=666, state/action=32
  phase micro-stage label_id values=0..13 for episode 0
  padding zero

两个 smoke repo 的 meta/rmbench/key_state_config.yaml 都是 resolved config，
没有额外的顶层 config wrapper 或 episode summary。
```

正式 conversion 验收：

```text
cover_blocks_demo_clean_state_key_state_label_id:
  episodes=50, frames=995-1054, total_frames=50904, state/action=32
  phase label_id values=0..5
  red/green/blue attribute label_id values valid, padding zero

battery_try_demo_clean_state_key_state_micro_stage_label_id:
  episodes=50, frames=454-882, total_frames=32626, state/action=32
  phase micro-stage label_id values=0..17
  padding zero

两个正式 repo 的 meta/rmbench/ 四个复现文件齐全。
source_data_config.yaml 和 source_data_command.txt 与 source demo metadata 逐字节一致。
key_state_config.yaml 是 resolved config，没有额外的顶层 config wrapper 或 episode summary。
```

## 训练

共用训练配置：

```text
policy/pi05/src/openpi/training/config.py: pi0_aloha_key_state_lora
```

正式训练前的 norm stats 使用 `--max-frames=10000` 计算，并覆盖写入对应 assets 目录：

```text
policy/pi05/assets/pi0_aloha_key_state_lora/cover_blocks_demo_clean_state_key_state_label_id/norm_stats.json
policy/pi05/assets/pi0_aloha_key_state_lora/battery_try_demo_clean_state_key_state_micro_stage_label_id/norm_stats.json
```

训练 run：

```text
cover_blocks:
  repo_id=cover_blocks_demo_clean_state_key_state_label_id
  exp_name=pi0_key_state_encoding_ablation_cover_blocks_label_id
  checkpoint=policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_encoding_ablation_cover_blocks_label_id
  stdout=logs/pi0_key_state_encoding_ablation/train_cover_blocks_label_id_gpu1.log
  wandb=4de2ci40
  status=running

battery_try:
  repo_id=battery_try_demo_clean_state_key_state_micro_stage_label_id
  exp_name=pi0_key_state_encoding_ablation_battery_try_micro_stage_label_id
  checkpoint=policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_encoding_ablation_battery_try_micro_stage_label_id
  stdout=logs/pi0_key_state_encoding_ablation/train_battery_try_micro_stage_label_id_gpu2.log
  wandb=16hwffku
  status=running
```
