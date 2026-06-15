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
