# Pi0.5 Multi-task Shared-Memory Representation

本实验组在五个任务上比较 dense full key state 与 serial-soft state token。此前完成的
`rearrange_blocks` 与本批次新增的四个任务统一归入同一个 W&B group：

| Task | Semantic memory |
| --- | --- |
| rearrange_blocks | phase(3) + empty_mat_side(3) + button_press_status(3) |
| put_back_block | phase(3) + origin_mat(5) |
| swap_blocks | phase(4) + initial_empty_tray(4) + first_origin_tray(4) |
| battery_try | phase(4) |
| cover_blocks | phase(6) + red/green/blue_pos(4 each) |

每个任务只有一份语义 schema 和一份 LeRobot repo。数据集同时保存 32D dense
state/action 与离散 token sidecar。Full 路径读取完整 32D；serial 路径在 normalization
前裁为前 14D robot state/action，并读取 token sidecar。

## 控制变量

- train seed: 0
- batch size: 32
- train steps: 30,000
- action horizon: 50
- token query stride: 20
- W&B project/group: `RMBench` / `pi05_multitask_shared_memory_representation`
- full train config: 五个任务均为 `pi05_full_key_state`
- serial train config: 新四任务为 `pi05_multitask_state_token_serial_soft`；历史 rearrange run 为
  `pi05_rearrange_state_token_boundary_ablation` 加 `key_state_token_mode=serial`

两个 serial config 的有效模型结构、state loss、优化器、teacher forcing 和 soft boundary
相同；历史 rearrange config 额外携带仅供 hard boundary 使用的 guard sidecar、旧 norm asset
路径和实验 metadata。新四任务仅通过 `repo_id`、run name 和 token category counts 覆盖配置，
不新增八份 train config。

## 数据

| Task | Shared LeRobot repo |
| --- | --- |
| put_back_block | `put_back_block_demo_clean_state_shared_memory` |
| swap_blocks | `swap_blocks_demo_clean_state_shared_memory` |
| battery_try | `battery_try_demo_clean_state_shared_memory` |
| cover_blocks | `cover_blocks_demo_clean_state_shared_memory` |

语义 schema 位于 `converter_configs/memory_schemas/<task>.yaml`，adapter 位于
`converter_configs/shared_memory/<task>.yaml`。

从 workspace 根目录转换：

```bash
cd policy/pi05/examples/aloha_real
for task in put_back_block swap_blocks battery_try cover_blocks; do
  ../../.venv/bin/python convert_robotwin_key_state_to_lerobot.py \
    --config ../../../../converter_configs/shared_memory/${task}.yaml
done
```

四个 shared repo 从 clean commit `9da5cee` 转换，帧数依次为
17,588 / 29,920 / 32,626 / 50,904。与既有 token/full 数据逐帧审计后，机器人
state/action、token input/target/mask 的差异元素均为 0。因此 full 复用同一 32D dense
数据的既有统计量，serial 复用同一 14D robot 数据的既有统计量；复制后已校验维度为
32/32 与 14/14。

## 正式训练

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_shared_memory_representation/jobs_train.json \
  --gpus 1,2,3,4,5,6,7 \
  --state policy/pi05/checkpoints/pi05_multitask_shared_memory_representation_queue_state.json
```

共 8 条训练。最初在 GPU1-7 启动 7 条；经确认允许使用 GPU0 后，停止原协调进程（不影响已独立运行的训练进程），并在 GPU0 启动最后一条。

正式队列于 2026-08-13 14:47 CST 从 clean commit `9da5cee` 启动：

此前完成的 rearrange runs 已于 2026-08-13 迁入同一 W&B group：full `fw5tgans`、
serial-soft `zs2c9uzy`。因此该 group 当前包含五任务共 10 个模型。

| Run | GPU | W&B ID | 启动状态 |
| --- | ---: | --- | --- |
| put_back_block_full_key_state_seed0 | 1 | `wecc9fjx` | running |
| put_back_block_serial_soft_seed0 | 2 | `qdrni4cb` | running |
| swap_blocks_full_key_state_seed0 | 3 | `1gaddz2j` | running |
| swap_blocks_serial_soft_seed0 | 4 | `3dl52sms` | running |
| battery_try_full_key_state_seed0 | 5 | `4bejv16r` | running |
| battery_try_serial_soft_seed0 | 6 | `a72dsvz0` | running |
| cover_blocks_full_key_state_seed0 | 7 | `9416n4r2` | running |
| cover_blocks_serial_soft_seed0 | 0 | `r5bu3qoc` | running |

目标 checkpoint：

```text
policy/pi05/checkpoints/pi05_full_key_state/<task>_full_key_state_seed0/30000
policy/pi05/checkpoints/pi05_multitask_state_token_serial_soft/<task>_serial_soft_seed0/30000
```

## 状态

- shared schema / converter config: implemented
- converter regression: 9 passed
- related non-manual regression: 18 passed, 2 deselected
- one-episode conversion smoke: 4/4 passed
- full dataset conversion: 4/4 passed; 131,038 total frames
- old/new full-frame equality audit: passed; all compared fields have 0 mismatched elements
- norm stats: ready and dimension-checked for 8/8 model/task combinations
- 2-step, bs=32 train smoke: 8/8 passed
- formal 30k training: 8 running, 0 queued; 0 failed
