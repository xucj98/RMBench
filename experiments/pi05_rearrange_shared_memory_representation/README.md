# Pi0.5 Rearrange Blocks Shared-Memory Comparison

本批次比较 rearrange_blocks 上的两种 key-state 使用方式：

- `full_key_state`：把共享 memory 编码成 32D observation/action 中的 one-hot dense tail；
- `serial_soft`：预测离散 state token，再用当前 token 条件化 action，允许 action chunk 跨 phase。

两组使用同一个 LeRobot repo、同一组三字段语义和同一批机器人轨迹。该实验仍同时包含
joint/per-step 与 serial/per-query 的结构差异，因此用于比较当前两套完整方案，不宣称是纯编码消融。

## Shared memory

唯一语义 schema：

`converter_configs/memory_schemas/rearrange_blocks.yaml`

字段：

- `phase`：3 类；
- `empty_mat_side`：unknown/left/right；
- `button_press_status`：NA/unconfirmed/confirmed。

唯一 converter config：

`converter_configs/shared_memory/rearrange_blocks.yaml`

唯一 LeRobot repo：

`rearrange_blocks_demo_clean_state_shared_memory`

dataset 同时保存 32D dense full-key state/action 和离散 token sidecar。Full-key transform 使用完整
32D；state-token transform 只读取前 14D robot state/action，因此 dense memory 不进入 token
模型的连续输入、action loss 或 norm stats。

## 数据准备

从 workspace 根目录运行：

```bash
cd policy/pi05/examples/aloha_real && ../../.venv/bin/python \
  convert_robotwin_key_state_to_lerobot.py \
  --config ../../../../converter_configs/shared_memory/rearrange_blocks.yaml

cd policy/pi05 && PYTHONPATH=src .venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_full_key_state \
  --repo-id rearrange_blocks_demo_clean_state_shared_memory \
  --max-frames 10000
```

Serial-token 复用已有 `rearrange_blocks_state_token` norm stats：shared dataset 的前 14D
robot state/action 与原 token dataset 相同，且 token transform 会在 normalization 前裁掉 dense tail。

## 正式训练

- batch size：32
- steps：30,000
- train seed：0
- action horizon：50
- token query stride：20
- W&B project/group：`RMBench` / `pi05_rearrange_shared_memory_representation`

两个 seed 0 历史 run、seed 42 复现 run、random-lag seed 0 run 以及对应 eval 均归入这个 rearrange 专用 group。
曾短暂迁入的 `pi05_multitask_shared_memory_representation` group 已纠正；run ID、指标和
checkpoint 不变。

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_shared_memory_representation/jobs_train.json \
  --gpus 1,2 \
  --state policy/pi05/checkpoints/pi05_rearrange_shared_memory_representation_queue_state.json
```

产物：

- `policy/pi05/checkpoints/pi05_full_key_state/shared_memory_full_key_state_seed0/30000`
- `policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/shared_memory_serial_soft_seed0/30000`

状态：两组训练均已完成，30k checkpoint 完整保存，训练进程返回码均为 0。

## Seed 0、pi0_step=30 评测

两组均使用 `demo_clean_eval`、eval seed 0、100 rollouts。模型仍输出 50-step action
chunk，但每次只执行前 30 步后重新 query；每组前 5 条 rollout 保存带 key-state overlay
的视频。

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_shared_memory_representation/jobs_eval_seed0_step30.json \
  --gpus 1,2 \
  --state eval_result/pi05_rearrange_shared_memory_representation/_step30_seed0_queue_state.json
```

结果目录：

- `eval_result/pi05_rearrange_shared_memory_representation/full_key_state_seed0@ckpt30k_step30_100ep_seed0`
- `eval_result/pi05_rearrange_shared_memory_representation/serial_soft_seed0@ckpt30k_step30_100ep_seed0`

## Seed 42 serial-soft 复现

为区分 shared-memory 改造与训练随机种子的影响，补跑与旧实验一致的 train seed 42：

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_shared_memory_representation/jobs_train_seed42.json \
  --gpus 7 \
  --state policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/_formal_logs/shared_memory_serial_soft_seed42/queue_state.json
```

产物：

- `policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/shared_memory_serial_soft_seed42/30000`
- W&B run：`81pfcs8v`

状态：30k 训练已于 2026-08-14 23:15 完成，checkpoint 完整；step30 的 eval seed0/seed1 已于 2026-08-15 16:50 启动。

评测 manifests：

- `jobs_eval_seed42_step30_seed0.json`
- `jobs_eval_seed42_step30_seed1.json`

正式评测：

- eval seed0：GPU5，W&B `a2l344jr`，running；
- eval seed1：GPU6，W&B `74ry0mk8`，running。

## Random previous-state lag（15–50）

固定 `t-20` 会让 Serial 模型只看到一种 previous-state age。新增训练时动态采样：每次读取样本时
从闭区间 `[15, 50]` 均匀采样整数 lag，并取同 episode 的 `state_target[t-lag]`；若历史不足，
回退到 episode 初始 state。该变体继续使用同一个 shared-memory LeRobot dataset 和 norm stats，
不改变 state target、action supervision 或 50-step action horizon。推理时也不随机：仍使用上一次
实际 query 输出的 state，因此 `pi0_step` 决定真实 memory age。

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_shared_memory_representation/jobs_train_random_prev15_50_seed0.json \
  --gpus 0 \
  --state policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/_formal_logs/shared_memory_serial_soft_random_prev15_50_seed0/queue_state.json
```

产物：

- `policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/shared_memory_serial_soft_random_prev15_50_seed0/30000`

状态：2-step 真数据 smoke 已通过；2026-08-14 17:02 已在 GPU0 启动 30k 正式训练。
W&B run：`hpwjlw42`；训练代码提交：`8c00b3f`。

## Serial oracle-state validation

`state_token_rollout_mode=oracle` 时，模型仍输出预测 state 供诊断，但当前 action condition 和
下一次 query 的 previous-state memory 都使用环境 GT state。GT 只来自当前仿真物理状态，不读取
未来 expert trajectory。正式评测沿用 seed 0、100 rollouts、`pi0_step=30`：

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_shared_memory_representation/jobs_eval_oracle_seed0_step30.json \
  --gpus 6 \
  --state eval_result/pi05_rearrange_shared_memory_representation/_oracle_step30_seed0_queue_state.json
```

结果目录：

- `eval_result/pi05_rearrange_shared_memory_representation/serial_soft_seed0_oracle@ckpt30k_step30_100ep_seed0`

## Full key-state oracle validation

`key_state_rollout_mode=oracle` 时，每个环境观测写入模型前，按照 checkpoint 保存的 shared-memory
schema 将当前环境 GT 编码到 32D state 的 dense memory tail。模型输出的 32D action 仍只把前 14D
发送给机器人；预测 memory tail 不再递推到下一观测。chunk 内每个执行步后的观测也重新同步 GT，
不读取未来轨迹、不改变 50-step action horizon，评测仍只执行前 30 步。

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_shared_memory_representation/jobs_eval_full_oracle_seed0_step30.json \
  --gpus 6 \
  --state eval_result/pi05_rearrange_shared_memory_representation/_full_oracle_step30_seed0_queue_state.json
```

GPU6 与已有任务共享，manifest 固定 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`。结果目录：

- `eval_result/pi05_rearrange_shared_memory_representation/full_key_state_seed0_oracle@ckpt30k_step30_100ep_seed0`

当前结果：

| 模型 | rollout state | train seed | eval seed | success |
| --- | --- | ---: | ---: | ---: |
| full key state | predicted | 0 | 0 | 93/100 |
| full key state | oracle | 0 | 0 | 82/100 |
| serial-soft | predicted | 0 | 0 | 36/100 |
| serial-soft | oracle | 0 | 0 | 79/100 |

Full oracle 相比 full predicted 从 93/100 降到 82/100。两者第一次放置均为 100/100；
有效按钮按压从 94/100 降到 87/100，第二次放置从 93/100 降到 82/100。Oracle 的
18 次失败中 13 次发生在按钮动作、5 次发生在第二次搬运。因此 dense full 的环境 GT
不是性能上界：action tail 中递推的 memory 更像与动作 chunk 对齐的计划/阶段时钟，改为
物理确认后的 current-state GT 会改变闭环时序和输入分布。
