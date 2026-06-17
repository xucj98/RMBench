# cover_blocks_key_state_design

本批次用于验证 `cover_blocks` 的 key-state 语义设计是否是当前成功率为 0 的主要瓶颈。

已有结果显示：

```text
pi0_lora_baseline:             1/100 = 1%
pi0_lora_key_state 20000 step: 0/100 = 0%
pi0_lora_key_state 30000 step: 0/100 = 0%
Paper Mem-0:                   68%
Paper Pi0.5:                   0%
```

设计文档见：

```text
docs/cover_blocks_key_state_design.md
```

## 对比项

本批次先跑两个新数据处理变体：

```text
exec2_attr3_no_phase:
  operation/slot label-id + open_order attributes label-id。
  不显式提供 phase，检查二维执行状态加紧凑记忆是否足够。

phase_exec2_attr3:
  phase label-id + operation/slot label-id + open_order attributes label-id。
  检查显式 progress 信号是否仍然有帮助。
```

它们都复用现有 state-augmented source dataset：

```text
data/cover_blocks/demo_clean_state
```

source dataset 的生成配置和命令以该目录下的 `metadata/config.yaml` 和
`metadata/command.txt` 为准。

## LeRobot 转换

converter configs：

```text
experiments/cover_blocks_key_state_design/converter_configs/cover_blocks_exec2_attr3_no_phase.yaml
experiments/cover_blocks_key_state_design/converter_configs/cover_blocks_phase_exec2_attr3.yaml
```

目标 repo：

```text
exec2_attr3_no_phase:
  cover_blocks_demo_clean_state_key_state_exec2_attr3_no_phase

phase_exec2_attr3:
  cover_blocks_demo_clean_state_key_state_phase_exec2_attr3
```

当前状态：

```text
smoke conversion: passed
formal conversion: passed
```

smoke conversion 验收：

```text
exec2_attr3_no_phase:
  episodes=1, frames=1008, state/action=32
  operation label_id values=0..1
  slot label_id values=0..2
  open_slot_0/1/2 attribute label_id values valid and latched after acquisition window
  uncover slot labels match task_facts.open_order_positions
  padding dim 19:32 zero

phase_exec2_attr3:
  episodes=1, frames=1008, state/action=32
  phase label_id values=0..5
  operation label_id values=0..1
  slot label_id values=0..2
  open_slot_0/1/2 attribute label_id values valid and latched after acquisition window
  uncover slot labels match task_facts.open_order_positions
  padding dim 20:32 zero
```

formal conversion 验收：

```text
exec2_attr3_no_phase:
  repo_id=cover_blocks_demo_clean_state_key_state_exec2_attr3_no_phase
  episodes=50, frames=995-1054, total_frames=50904
  state/action=32
  operation/slot/open_slot_0/1/2 label_id values valid
  uncover slot labels match task_facts.open_order_positions
  open_slot_0/1/2 latch after acquisition window
  padding dim 19:32 zero
  meta/rmbench 四个复现文件齐全

phase_exec2_attr3:
  repo_id=cover_blocks_demo_clean_state_key_state_phase_exec2_attr3
  episodes=50, frames=995-1054, total_frames=50904
  state/action=32
  phase/operation/slot/open_slot_0/1/2 label_id values valid
  uncover slot labels match task_facts.open_order_positions
  open_slot_0/1/2 latch after acquisition window
  padding dim 20:32 zero
  meta/rmbench 四个复现文件齐全
```

## 训练

训练复用现有配置：

```text
policy/pi05/src/openpi/training/config.py: pi0_aloha_key_state_lora
```

正式训练前使用 `max_frames=10000` 计算 norm stats，并写入：

```text
policy/pi05/assets/pi0_aloha_key_state_lora/cover_blocks_demo_clean_state_key_state_exec2_attr3_no_phase/norm_stats.json
policy/pi05/assets/pi0_aloha_key_state_lora/cover_blocks_demo_clean_state_key_state_phase_exec2_attr3/norm_stats.json
```

训练 run：

```text
exec2_attr3_no_phase:
  status=running
  repo_id=cover_blocks_demo_clean_state_key_state_exec2_attr3_no_phase
  exp_name=cover_blocks_key_state_design_exec2_attr3_no_phase
  gpu=3
  checkpoint=policy/pi05/checkpoints/pi0_aloha_key_state_lora/cover_blocks_key_state_design_exec2_attr3_no_phase
  stdout=logs/cover_blocks_key_state_design/train_exec2_attr3_no_phase_gpu3.log
  wandb=xtwaal6k

phase_exec2_attr3:
  status=running
  repo_id=cover_blocks_demo_clean_state_key_state_phase_exec2_attr3
  exp_name=cover_blocks_key_state_design_phase_exec2_attr3
  gpu=5
  checkpoint=policy/pi05/checkpoints/pi0_aloha_key_state_lora/cover_blocks_key_state_design_phase_exec2_attr3
  stdout=logs/cover_blocks_key_state_design/train_phase_exec2_attr3_gpu5.log
  wandb=7l2as0ip
```

W&B project 使用 `RMBench`，group 使用 `cover_blocks_key_state_design`。
