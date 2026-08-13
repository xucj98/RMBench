# Pi0.5 Multi-task Shared-Memory Representation

本批次把 rearrange_blocks 的共享 memory schema / 单 LeRobot 数据集框架迁移到四个任务，
比较 dense full key state 与 serial-soft state token：

| Task | Semantic memory |
| --- | --- |
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
- full train config: `pi05_full_key_state`
- serial train config: `pi05_multitask_state_token_serial_soft`

四个任务仅通过 `repo_id`、run name 和 token category counts 覆盖配置，不新增八份
train config。

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

Full 与 serial 分别按各自进入模型的有效维度计算 norm stats，均显式使用
`--max-frames 10000`。

## 正式训练

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_shared_memory_representation/jobs_train.json \
  --gpus 1,2,3,4,5,6,7 \
  --state policy/pi05/checkpoints/pi05_multitask_shared_memory_representation_queue_state.json
```

共 8 条训练。GPU0 按项目默认规则保留；第 8 条 job 在前一条完成后自动接续。

目标 checkpoint：

```text
policy/pi05/checkpoints/pi05_full_key_state/<task>_full_key_state_seed0/30000
policy/pi05/checkpoints/pi05_multitask_state_token_serial_soft/<task>_serial_soft_seed0/30000
```

## 状态

- shared schema / converter config: implemented
- schema regression: 9 passed
- one-episode conversion smoke: 4/4 passed
- full dataset / norm stats / train smoke / formal training: pending
