# pi0_key_state_baseline

本批次用于评估 pi0 + key state baseline。当前先落地 4 个任务：

```text
rearrange_blocks
swap_blocks
battery_try
cover_blocks
```

## 数据生成

正式数据使用 `task_config/demo_clean_state.yml`，输出目录为：

```text
data/<task>/demo_clean_state
```

每个正式数据目录下的 `metadata/config.yaml` 和 `metadata/command.txt`
由采集入口自动记录 resolved config、启动命令、git commit、cwd 和白名单环境变量。
README 只记录批次语义、路径和验收摘要，不手写这些复现字段。

当前状态：

```text
正式 50ep 数据生成完成。
```

验收结果：

```text
rearrange_blocks: 50 hdf5, 50 instructions, frames 398-409, micro_stages=9
swap_blocks:      50 hdf5, 50 instructions, frames 584-681, micro_stages=14
battery_try:      50 hdf5, 50 instructions, frames 455-883, micro_stages=2/3/4
cover_blocks:     50 hdf5, 50 instructions, frames 996-1055, micro_stages=28/29
```

所有 episode 均通过以下检查：

```text
scene_info episode 数为 50。
language_annotation episode 数为 50。
info.task_facts 包含当前任务所需字段。
info.micro_stages 非空、递增、frame range 非空且不超过 hdf5 action frame 数。
每个数据目录包含 metadata/config.yaml 和 metadata/command.txt。
```

## LeRobot 转换

正式 LeRobot repo 写入本机默认 LeRobot cache：

```text
~/.cache/huggingface/lerobot/<repo_id>
```

每个 repo 的 `meta/rmbench/` 下包含 `key_state_config.yaml`、`convert_command.txt`、
`source_data_config.yaml` 和 `source_data_command.txt`。转换命令、git commit、cwd
和白名单环境变量由 `convert_command.txt` 自动记录；README 不重复手写这些字段。

正式 repo：

```text
rearrange_blocks: repo_id=rearrange_blocks_demo_clean_state_key_state
swap_blocks:      repo_id=swap_blocks_demo_clean_state_key_state
battery_try:      repo_id=battery_try_demo_clean_state_key_state
cover_blocks:     repo_id=cover_blocks_demo_clean_state_key_state
```

当前状态：

```text
四个任务的正式 50ep LeRobot 转换完成。
```

验收结果：

```text
rearrange_blocks: 50 episodes, frames 397-408, state/action=32
swap_blocks:      50 episodes, frames 583-680, state/action=32
battery_try:      50 episodes, frames 454-882, state/action=32
cover_blocks:     50 episodes, frames 995-1054, state/action=32
```

所有正式 repo 均通过以下检查：

```text
meta/rmbench 四个复现文件齐全。
phase one-hot 校验通过。
attribute one-hot 校验通过。
padding zero 校验通过。
source_data_config.yaml 与 source data metadata/config.yaml 一致。
source_data_command.txt 与 source data metadata/command.txt 一致。
```
