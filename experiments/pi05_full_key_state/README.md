# pi05_full_key_state

本批次用于评估 Pi0.5 full finetune 在 key-state 数据上的性能，当前覆盖：

```text
rearrange_blocks
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
num_train_steps: 30000
gpu: one run per GPU
xla_mem_fraction: 0.95
wandb project: RMBench
wandb group: pi05_full_key_state
eval: pending
```

## 数据和 Assets

使用已有 LeRobot repo：

```text
rearrange_blocks: repo_id=rearrange_blocks_demo_clean_state_key_state
swap_blocks: repo_id=swap_blocks_demo_clean_state_key_state
battery_try: repo_id=battery_try_demo_clean_state_key_state
```

norm stats 已按 `full_key_state` 训练配置计算并写入：

```text
policy/pi05/assets/full_key_state/rearrange_blocks_demo_clean_state_key_state/norm_stats.json
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
| `rearrange_blocks` | `rearrange_blocks_demo_clean_state_key_state` | 6 | running | `pdsniduh` | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_rearrange_blocks` | `policy/pi05/logs/full_key_state/pi05_full_key_state_rearrange_blocks.stdout.log` |
| `swap_blocks` | `swap_blocks_demo_clean_state_key_state` | 0 | running | `65ldopff` | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_swap_blocks` | `policy/pi05/logs/full_key_state/pi05_full_key_state_swap_blocks.stdout.log` |
| `battery_try` | `battery_try_demo_clean_state_key_state` | 7 | running | `2mueiwio` | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_battery_try` | `policy/pi05/logs/full_key_state/pi05_full_key_state_battery_try.stdout.log` |

训练命令、git commit、cwd、环境变量和 resolved train config 由 checkpoint metadata
自动记录；README 不重复手写这些字段。

`swap_blocks` 和 `battery_try` 启动时使用的是旧 commit 下的 `full_key_state`
配置，训练步数为 20000。`rearrange_blocks` 起使用 30000-step 默认配置。

## 评测

### Rearrange-blocks 30k 复现与执行步数消融

删除原30k checkpoint 后，按其 metadata 在训练 commit
`eddfff7e9ba0a33ce07fdc3833dc3f29f5ede458` 上重训
`rearrange_blocks_repro_eddfff7_seed42`。模型 action horizon 固定为50；这里只改变闭环中每次
实际执行的前 N 步 `pi0_step`，不改变模型结构或训练监督。

step50 已完成：eval seed0 为38/100，seed1 为46/100。追加 step15/20/30，两个 eval seed
各100 episodes，共6项。为与 step50 严格对齐，六项从干净的共享代码快照
`/mnt/public3/xcj/RMBench`、eval commit
`cbfbbc1c3b4b96b80499d7c24c84b244c7600a68` 启动；数据、checkpoint、task config 和
评测参数均不变。

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_full_key_state/jobs_repro_move_steps_15_20_30.json \
  --gpus 1,2,3,5,6,7 \
  --state eval_result/pi05_full_key_state_repro/_move_steps_15_20_30_queue_state.json
```

结果目录：

```text
eval_result/pi05_full_key_state_repro/
  rearrange_blocks_repro_eddfff7_seed42@ckpt30k_step<N>_100ep_seed<S>/
```

状态：step50 completed；step15/20/30 running。
