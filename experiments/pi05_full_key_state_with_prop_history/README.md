# pi05_full_key_state_with_prop_history

本批次验证 Pi0.5 full finetune 在 key-state 任务中加入 proprioceptive history 后的效果。
图像仍为当前单帧；state 输入为过去 3 帧加当前帧，共 4 帧，不使用 future state、推理延迟补偿或异步执行。

## 实验范围

```text
policy: pi05 full finetune
train_config_name: pi05_full_key_state_with_prop_history
tasks: rearrange_blocks, put_back_block
history: state_history_size=3, state_future_size=0, state_step=1
batch_size: 32
num_train_steps: 30000
save_interval: 10000
xla_mem_fraction: 0.95
wandb project: RMBench
wandb group: pi05_full_key_state_with_prop_history
wandb mode: online
train commit: 0f60a68c2deffd8ebc9f07df8122c1d00f50ac1f
eval: pending
```

模型保留 Pi0.5 原有的当前 state 离散 prefix，并把完整 4 帧 state sequence 经独立投影送入 action-expert suffix。
LeRobot 在 episode 起点对负时间偏移自动 padding；仿真推理在 episode 起点用第一帧填满 history buffer。

## 数据与 norm stats

复用已有 key-state LeRobot 数据，不重新采集 demo：

```text
rearrange_blocks_demo_clean_state_key_state
put_back_block_demo_clean_state_key_state
```

`put_back_block` 的 LeRobot cache 如不存在，先从 workspace 根目录恢复：

```bash
cd policy/pi05 && PYTHONPATH=src .venv/bin/python \
  examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
  --config ../../converter_configs/key_state_baseline/put_back_block.yaml
```

每个 repo 使用至多 10000 frames 重新计算 norm stats：

```bash
bash experiments/pi05_full_key_state_with_prop_history/commands/prepare_norm_stats.sh
```

产物：

```text
policy/pi05/assets/pi05_full_key_state_with_prop_history/<repo_id>/norm_stats.json
```

本次 10000-frame 统计文件的 SHA-256：

```text
rearrange_blocks: 06377a8d777eb66492148aab2b28451a5fb1fa36a7c008699c50f1490f855f3b
put_back_block:   550fb4810069fef006794e23d799d8ee1505999601efc0b145a47fc1f1fa5488
```

## 训练

从 workspace 根目录启动；GPU 作为运行参数传入，默认不要使用 GPU0：

```bash
bash experiments/pi05_full_key_state_with_prop_history/commands/train.sh rearrange_blocks 1
bash experiments/pi05_full_key_state_with_prop_history/commands/train.sh put_back_block 2
```

产物与日志：

```text
policy/pi05/checkpoints/pi05_full_key_state_with_prop_history/<task>/<step>
policy/pi05/checkpoints/pi05_full_key_state_with_prop_history/<task>/train.stdout.log
policy/pi05/checkpoints/pi05_full_key_state_with_prop_history/<task>/train.pid
```

| Task | Repo ID | Status | GPU | wandb id | Checkpoint |
| --- | --- | --- | ---: | --- | --- |
| `rearrange_blocks` | `rearrange_blocks_demo_clean_state_key_state` | running (PID 401514) | 1 | [`iy84omxw`](https://wandb.ai/xucj98/RMBench/runs/iy84omxw) | `policy/pi05/checkpoints/pi05_full_key_state_with_prop_history/rearrange_blocks` |
| `put_back_block` | `put_back_block_demo_clean_state_key_state` | running (PID 401520) | 2 | [`fig03zgh`](https://wandb.ai/xucj98/RMBench/runs/fig03zgh) | `policy/pi05/checkpoints/pi05_full_key_state_with_prop_history/put_back_block` |

## 评测

```text
eval_result/pi05_full_key_state_with_prop_history/<task>@ckpt30k
```

训练于 2026-08-02 启动，当前尚未启动评测；训练完成后补充评测命令和结果摘要。

### Prop-history 时序审计

训练和评测统一使用从旧到新的 4 帧 state sequence：

```text
[s(t-3), s(t-2), s(t-1), s(t)]
```

训练侧由 LeRobot `delta_timestamps=[-3, -2, -1, 0] / fps` 取样，episode 起点的
负时间索引 clip 到第 0 帧。评测侧在首次 query 用当前 state 填满 4 个位置；chunk
执行时，每执行一个 `a(t) -> s(t+1)`，先用该 action 更新 key state，再 append
post-action proprio state。chunk 最后一步的 post-action state 由下一次 outer query
append，且 chunk 内不会提前 append，因此不会遗漏或重复。

例如执行 50 步后，下一次 query 的输入为：

```text
[s(t+47), s(t+48), s(t+49), s(t+50)]
```

这与训练侧在 frame `t+50` 的取样完全一致。图像仍只使用 query 时刻的当前帧；chunk
内更新 observation window 只是维护 proprio history，不额外引入图像历史。
