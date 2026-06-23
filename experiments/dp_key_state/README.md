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
job_type: train / eval
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

正式 zarr 转换和训练已完成，5 个正式 eval 已启动。

| Task | zarr frames | GPU | train | wandb | checkpoint | stdout |
| --- | ---: | ---: | --- | --- | --- | --- |
| `put_back_block` | 17588 | 1 | finished | `ab976la2` | `policy/DP/checkpoints/put_back_block-demo_clean_state_key_state-50-0/600.ckpt` | `policy/DP/checkpoints/put_back_block-demo_clean_state_key_state-50-0/stdout.log` |
| `rearrange_blocks` | 20103 | 2 | finished | `vf8hbz9z` | `policy/DP/checkpoints/rearrange_blocks-demo_clean_state_key_state-50-0/600.ckpt` | `policy/DP/checkpoints/rearrange_blocks-demo_clean_state_key_state-50-0/stdout.log` |
| `swap_blocks` | 29920 | 3 | finished | `3poajv2a` | `policy/DP/checkpoints/swap_blocks-demo_clean_state_key_state-50-0/600.ckpt` | `policy/DP/checkpoints/swap_blocks-demo_clean_state_key_state-50-0/stdout.log` |
| `battery_try` | 32626 | 4 | finished | `hcuznegt` | `policy/DP/checkpoints/battery_try-demo_clean_state_key_state-50-0/600.ckpt` | `policy/DP/checkpoints/battery_try-demo_clean_state_key_state-50-0/stdout.log` |
| `cover_blocks` | 50904 | 5 | finished | `cia97huy` | `policy/DP/checkpoints/cover_blocks-demo_clean_state_key_state-50-0/600.ckpt` | `policy/DP/checkpoints/cover_blocks-demo_clean_state_key_state-50-0/stdout.log` |

正式 eval 使用 100 rollouts，前 5 个 rollout 录制 key-state overlay 视频。

| Task | GPU | eval | wandb | eval result | stdout |
| --- | ---: | --- | --- | --- | --- |
| `put_back_block` | 0 | running | `tbrnpcau` | `eval_result/dp_key_state/put_back_block` | `eval_result/dp_key_state/put_back_block/stdout.log` |
| `rearrange_blocks` | 1 | running | `9w3uoco5` | `eval_result/dp_key_state/rearrange_blocks` | `eval_result/dp_key_state/rearrange_blocks/stdout.log` |
| `swap_blocks` | 2 | running | `1ahmwrzb` | `eval_result/dp_key_state/swap_blocks` | `eval_result/dp_key_state/swap_blocks/stdout.log` |
| `battery_try` | 3 | running | `hjq06k6l` | `eval_result/dp_key_state/battery_try` | `eval_result/dp_key_state/battery_try/stdout.log` |
| `cover_blocks` | 4 | running | `57dkr449` | `eval_result/dp_key_state/cover_blocks` | `eval_result/dp_key_state/cover_blocks/stdout.log` |

每个 zarr 目录下的 `meta/rmbench/` 均包含：

```text
key_state_config.yaml
convert_command.txt
source_data_config.yaml
source_data_command.txt
summary.yaml
```

`convert_command.txt` 显示正式转换在 clean worktree 上执行。
