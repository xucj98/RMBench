# Pi0.5 Rearrange Blocks State-Token / Boundary Ablation

本批次在 rearrange_blocks 上比较两个正交设计轴，并补充 action chunk 实际执行步数
（pi0_step）消融：

| Cell | State/action 结构 | Action boundary | exp_name |
| --- | --- | --- | --- |
| P-S | Parallel | Soft | parallel_soft_seed42 |
| P-H | Parallel | Hard | parallel_hard_seed42 |
| S-S | Serial | Soft | serial_soft_seed42 |
| S-H | Serial | Hard | serial_hard_seed42 |

四组统一使用 Pi0.5 full finetune、batch size 32、30,000 steps、seed 42、action horizon 50、
query stride 20。模型和边界定义详见
[设计文档](../../docs/rearrange_blocks_state_token_boundary_design.md)。

四组属于同一个 exp group，并共用 train config
pi05_rearrange_state_token_boundary_ablation。Parallel/Serial 和 Soft/Hard 只通过正式命令的
CLI override 指定，不复制配置对象。

## 数据

数据集 repo id 为 rearrange_blocks_demo_clean_state_token。机器人 state/action 保持纯 14 维，
三字段离散 sidecar 为：

    phase: P0 / P1 / P2
    empty_mat_side: unknown / left / right
    button_press_status: NA / unconfirmed / confirmed

state_input[t] = state_target[t-20]，episode 开头固定为 (P0, unknown, NA)。现有 HDF5
没有按钮关节或接触事件；本批次暂用 language_annotation.json 第 4 个零基 segment
（press-down）的末帧定位 guard。该边界是示范程序分段标注，不是物理接触真值，必须作为
结果解释限制保留。converter 配置在
[converter config](../../converter_configs/state_token/rearrange_blocks.yaml)。

Soft/Hard 使用同一个 LeRobot 数据集和同一份 rearrange_blocks_state_token norm stats。
stats 按原始连续 action chunk（Soft）计算；Hard 仅在训练 transform 内改写跨边界的 action
supervision，避免归一化尺度成为额外变量。

## 训练与产物

- 正式 checkpoint：policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/<exp_name>/<step>
- W&B project：RMBench
- W&B group：pi05_rearrange_state_token_boundary_ablation
- smoke 使用相同 batch size，但只训练 2 steps，不进入正式结果。
- runner 状态和 stdout 位于对应 ignored checkpoint 或 eval result 目录。
- 每个 checkpoint 的 metadata/ 保存训练命令、commit、配置和数据 provenance。
- 每个评测目录的 config.yaml、command.txt 和 wandb_id.txt 保存评测 provenance。

训练入口：

    python script/run_job_queue.py \
      --jobs experiments/pi05_rearrange_state_token_boundary_ablation/jobs_formal.json \
      --gpus 3,4 \
      --state policy/pi05/checkpoints/pi05_rearrange_state_token_boundary_ablation/formal_queue_state.json

正式训练四组均已完成至 30k。训练时使用过 GPU3/4 和 GPU6/7；GPU 分配只是运行资源，
不是实验变量。

## 评测设置

所有正式结果使用：

    task: rearrange_blocks / demo_clean_eval
    checkpoint: 30000（另有两项 20k 早期检查）
    test_num: 100
    eval seed: 0（episode seed 100000..100099）
    instruction_type: unseen
    model action_horizon: 50（不修改 checkpoint 或模型输出长度）
    videos: 5，包含 key-state overlay

pi0_step=N 表示每次仍预测 50-step action chunk，但只执行前 N 步便重新 query。因而
step15/20/30/50 只改变闭环 replanning 频率，不改变训练监督和模型结构。

评测结果统一位于：

    eval_result/pi05_rearrange_state_token_boundary_ablation/
      <exp_name>@ckpt30k_100ep_seed0/                 # step50
      <exp_name>@ckpt30k_step<N>_100ep_seed0/         # step15/20/30

## 已完成结果

成功率如下。20k 只评测过两个 Parallel 变体，不纳入完整 2×2 结论。

| Variant | 20k / step50 | 30k / step50 | 30k / step20 | 30k / step15 | step20 vs. step50 | step15 vs. step20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parallel Soft | 19% | 15% | 6% | 3% | -9 pp | -3 pp |
| Parallel Hard | 30% | 15% | 27% | 12% | +12 pp | -15 pp |
| Serial Soft | — | 38% | 35% | 23% | -3 pp | -12 pp |
| Serial Hard | — | 21% | 38% | 37% | +17 pp | -1 pp |

step20 与 step50 使用完全相同的 100 个 episode seed。逐 episode 配对如下：

| Variant | 两者均成功 | 仅 step50 成功 | 仅 step20 成功 | 两者均失败 | exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parallel Soft | 0 | 15 | 6 | 79 | 0.078 |
| Parallel Hard | 2 | 13 | 25 | 60 | 0.073 |
| Serial Soft | 16 | 22 | 19 | 43 | 0.755 |
| Serial Hard | 8 | 13 | 30 | 49 | 0.014 |

step15 与 step20 的逐 episode 配对为：

| Variant | 两者均成功 | 仅 step20 成功 | 仅 step15 成功 | 两者均失败 | exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parallel Soft | 0 | 6 | 3 | 91 | 0.508 |
| Parallel Hard | 3 | 24 | 9 | 64 | 0.014 |
| Serial Soft | 9 | 26 | 14 | 51 | 0.081 |
| Serial Hard | 18 | 20 | 19 | 43 | 1.000 |

这里的 p 值是未做多重比较修正的同 seed 二项精确检验，只用于辅助判断翻转是否对称。
step15 相比 step20，Parallel Hard 的下降具有较清楚的单项配对证据；Serial Hard 则基本
持平。所有结论仍限于一个训练 seed。

### 失败模式

- Hard 模型的 button_not_released 随执行步数缩短继续减少：Parallel Hard 为
  20（step50）→ 5（step20）→ 2（step15），Serial Hard 为 21 → 0 → 0。
  更频繁观察确实改善了按钮释放/边界过冲。
- 但 Parallel Hard 的 first_placement_not_completed 从 step20 的 9 激增到 step15 的 38，
  使成功率从 27% 降至 12%。15 步开始破坏前段连续抓取/放置，而不只是改善按钮边界。
- Parallel Soft 的 button_pressed_multiple_times 为 2 → 31 → 19；step15 下重复按压仍远高于
  step50，同时 button_not_pressed 从 step20 的 33 回升到 51，最终仅 3% 成功。
- Serial Soft 随执行步数缩短持续下降（38% → 35% → 23%），主要瓶颈仍是
  button_not_pressed（61 / 54 / 61）。
- Serial Hard 对短执行 chunk 最稳健：step20 为 38%，step15 为 37%。总体上 step20 是目前
  更好的折中点；继续缩短到 15 没有带来额外收益。

## step15 / step30 追加消融

追加评测复用同一组 30k checkpoint 和上述 100 个 episode seed。八个 run 由一个 manifest
定义，四张 GPU 各先执行一个 step15 run，完成后自动接续 step30：

    setsid bash -lc 'python script/run_job_queue.py \
      --jobs experiments/pi05_rearrange_state_token_boundary_ablation/jobs_eval_move_steps_15_30.json \
      --gpus 0,1,2,3 \
      --state eval_result/pi05_rearrange_state_token_boundary_ablation/_move_steps_15_30_queue_state.json' \
      > eval_result/pi05_rearrange_state_token_boundary_ablation/_move_steps_15_30_queue.stdout.log 2>&1 &

step15 四组各 100 episodes 已完成，step30 四组正在运行。这组实验用于区分目前观察到的
交互是 step20 的偶然点效应，还是随 replanning 频率变化的趋势。step30 完成后应把
step15/20/30/50 四点补入上表，并优先比较：

    Hard: button_not_released 是否随 move steps 缩短而持续下降。
    Soft: button_pressed_multiple_times 是否随 move steps 缩短而持续上升。

## 状态

- structured 数据转换：completed
- static/core regression：completed
- 2×2 smoke：completed（bs=32，四组均 2/2 steps，return code 0）
- 2×2 formal training：completed（bs=32，30k）
- 30k / step50 evaluation：completed（四组各 100 episodes）
- 30k / step20 evaluation：completed（四组各 100 episodes）
- 30k / step15 evaluation：completed（四组各 100 episodes）
- 30k / step30 evaluation：running（四组并行）
