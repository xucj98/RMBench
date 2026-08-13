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

两个 seed 0 历史 run、新增的 seed 42 run 以及对应 eval 均归入这个 rearrange 专用 group。
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

状态：2026-08-13 已在 GPU7 启动，30k 训练中。

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

当前结果：

| 模型 | rollout state | train seed | eval seed | success |
| --- | --- | ---: | ---: | ---: |
| full key state | predicted | 0 | 0 | 93/100 |
| serial-soft | predicted | 0 | 0 | 36/100 |
| serial-soft | oracle | 0 | 0 | 79/100 |
