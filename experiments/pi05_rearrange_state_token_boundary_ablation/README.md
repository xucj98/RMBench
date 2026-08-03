# Pi0.5 Rearrange Blocks State-Token / Boundary Ablation

本批次在 `rearrange_blocks` 上比较两个正交设计轴：

| Cell | State/action 结构 | Action boundary | exp_name |
| --- | --- | --- | --- |
| P-S | Parallel | Soft | `parallel_soft_seed42` |
| P-H | Parallel | Hard | `parallel_hard_seed42` |
| S-S | Serial | Soft | `serial_soft_seed42` |
| S-H | Serial | Hard | `serial_hard_seed42` |

四组统一使用 Pi0.5 full finetune、batch size 32、30,000 steps、seed 42、action horizon 50、
query stride 20。模型和边界定义详见
[`docs/rearrange_blocks_state_token_boundary_design.md`](../../docs/rearrange_blocks_state_token_boundary_design.md)。

四组属于同一个 exp group，并共用 train config
`pi05_rearrange_state_token_boundary_ablation`。Parallel/Serial 和 Soft/Hard 只通过正式命令的
CLI override 指定，不复制配置对象。

## 数据

数据集 repo id 为 `rearrange_blocks_demo_clean_state_token`。机器人 state/action 保持纯 14 维，
三字段离散 sidecar 为：

```text
phase: P0 / P1 / P2
empty_mat_side: unknown / left / right
button_press_status: NA / unconfirmed / confirmed
```

`state_input[t] = state_target[t-20]`，episode 开头固定为 `(P0, unknown, NA)`。现有 HDF5
没有按钮关节或接触事件；本批次暂用 `language_annotation.json` 第 4 个零基 segment
（press-down）的末帧定位 guard。该边界是示范程序分段标注，不是物理接触真值，必须作为
结果解释限制保留。converter 配置在
[`converter_configs/state_token/rearrange_blocks.yaml`](../../converter_configs/state_token/rearrange_blocks.yaml)。

Soft/Hard 使用同一个 LeRobot 数据集和同一份 `rearrange_blocks_state_token` norm stats。
stats 按原始连续 action chunk（Soft）计算；Hard 仅在训练 transform 内改写跨边界的 action
supervision，避免归一化尺度成为额外变量。

## 训练与产物

- 正式 checkpoint：`policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/<exp_name>/<step>`
- W&B project：`RMBench`
- W&B group：`pi05_rearrange_state_token_boundary_ablation`
- smoke 使用相同 batch size，但只训练 2 steps，不进入正式结果。
- runner 状态和 stdout 位于对应 ignored checkpoint 目录。

GPU 资源不写入实验变体。当前批次按用户指定只使用物理 GPU 3、4：

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_state_token_boundary_ablation/jobs_smoke.json \
  --gpus 3,4 \
  --state policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/smoke_queue_state.json

python script/run_job_queue.py \
  --jobs experiments/pi05_rearrange_state_token_boundary_ablation/jobs_formal.json \
  --gpus 3,4 \
  --state policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/formal_queue_state.json
```

正式 manifest 的顺序是 P-S、P-H、S-S、S-H，因此 GPU3/4 首先各启动一个 Parallel
实验；任一 GPU 释放后，runner 自动启动后续 Serial 实验。

## 状态

- structured 数据转换：completed
- static/core regression：completed
- 2×2 smoke：completed（bs=32，四组均 2/2 steps，return code 0）
- 2×2 formal training：pending
- final evaluation：pending
