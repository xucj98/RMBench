# dp_key_state

本批实验验证 DP 在 5 个 key-state 任务上的表现。key-state 设计与
`pi05_full_key_state` 使用的 baseline schema 保持一致。

## 实验范围

任务：

```text
put_back_block
rearrange_blocks
swap_blocks
battery_try
cover_blocks
```

数据来源：

```text
data/<task>/demo_clean_state
```

公共 key-state schema：

```text
converter_configs/key_state_baseline/<task>.yaml
```

DP zarr 输出：

```text
policy/DP/data/<task>-demo_clean_state_key_state-50.zarr
```

DP checkpoint 输出：

```text
policy/DP/checkpoints/<task>-demo_clean_state_key_state-50-0/
```

wandb：

```text
project: RMBench
group: dp_key_state
job_type: train
```

## 运行方式

转换单个任务：

```bash
python policy/DP/process_key_state_data.py \
  --config converter_configs/key_state_baseline/<task>.yaml
```

训练单个任务：

```bash
cd policy/DP
bash train.sh <task> demo_clean_state_key_state 50 0 32 <gpu_id> dp_key_state
```

正式训练前先在 `_smoke` zarr 上做 conversion smoke 和 DP debug train smoke。
smoke 产物不进入本 README 的结果表，正式转换和训练从 clean commit 启动。

## 当前状态

正式 zarr 转换已完成，5 个训练已启动。

| Task | zarr frames | GPU | train | wandb | checkpoint | stdout |
| --- | ---: | ---: | --- | --- | --- | --- |
| `put_back_block` | 17588 | 1 | running | `ab976la2` | `policy/DP/checkpoints/put_back_block-demo_clean_state_key_state-50-0` | `policy/DP/checkpoints/put_back_block-demo_clean_state_key_state-50-0/stdout.log` |
| `rearrange_blocks` | 20103 | 2 | running | `vf8hbz9z` | `policy/DP/checkpoints/rearrange_blocks-demo_clean_state_key_state-50-0` | `policy/DP/checkpoints/rearrange_blocks-demo_clean_state_key_state-50-0/stdout.log` |
| `swap_blocks` | 29920 | 3 | running | `3poajv2a` | `policy/DP/checkpoints/swap_blocks-demo_clean_state_key_state-50-0` | `policy/DP/checkpoints/swap_blocks-demo_clean_state_key_state-50-0/stdout.log` |
| `battery_try` | 32626 | 4 | running | `hcuznegt` | `policy/DP/checkpoints/battery_try-demo_clean_state_key_state-50-0` | `policy/DP/checkpoints/battery_try-demo_clean_state_key_state-50-0/stdout.log` |
| `cover_blocks` | 50904 | 5 | running | `cia97huy` | `policy/DP/checkpoints/cover_blocks-demo_clean_state_key_state-50-0` | `policy/DP/checkpoints/cover_blocks-demo_clean_state_key_state-50-0/stdout.log` |

每个 zarr 目录下的 `meta/rmbench/` 均包含：

```text
key_state_config.yaml
convert_command.txt
source_data_config.yaml
source_data_command.txt
summary.yaml
```

`convert_command.txt` 显示正式转换在 clean worktree 上执行。
