# Pi0.5 Multi-task Serial + Soft State Token

本实验把 `rearrange_blocks` 上表现最好的 Serial + Soft state-token 方案迁移到四个 RMBench
任务：`put_back_block`、`swap_blocks`、`battery_try` 和 `cover_blocks`。

四个任务共用一个 train config：`pi05_multitask_state_token_serial_soft`。任务差异只通过
正式命令覆盖 `data.repo_id`、`model.key_state_num_values` 和 `exp_name`，不复制四份配置。

## 固定训练设置

- 模型：Pi0.5 full finetune
- state/action 结构：Serial（state token 作为 action prediction 的条件）
- action boundary：Soft（action chunk 允许跨 phase）
- batch size：32
- train steps：30,000
- seed：42
- action horizon：50
- state-token query stride：20
- norm stats：每个 repo 使用 `--max-frames 10000`
- W&B project/group：`RMBench` / `pi05_multitask_state_token_serial_soft`

## 数据与 token schema

机器人 `observation.state` 和 `action` 保持 14 维；离散状态通过 LeRobot sidecar 字段提供，
不进入连续向量和 norm stats。所有任务均使用：

```text
state_input[t] = state_target[t - 20]
episode 开头的 state_input = 全 0
serial training conditioning = teacher forcing
```

| Task | LeRobot repo id | Token fields | Category counts |
| --- | --- | --- | --- |
| put_back_block | `put_back_block_demo_clean_state_token` | phase, origin_mat | `(3, 5)` |
| swap_blocks | `swap_blocks_demo_clean_state_token` | phase, initial_empty_tray, first_origin_tray | `(4, 4, 4)` |
| battery_try | `battery_try_demo_clean_state_token` | phase | `(4,)` |
| cover_blocks | `cover_blocks_demo_clean_state_token` | phase, red_pos, green_pos, blue_pos | `(6, 4, 4, 4)` |

正式转换使用 clean commit `fbb2db7`，四个 repo 的总帧数依次为
17,588 / 29,920 / 32,626 / 50,904。norm stats 按项目规范显式使用
`--max-frames 10000`；bs=32 时实际统计 312 batches，即 9,984 frames。最终 SHA256：

| Task | norm_stats.json SHA256 |
| --- | --- |
| put_back_block | `e99a6abe7904f7c4265a56c487d97302bf32bd41fb847aad211c1f857603a729` |
| swap_blocks | `acb30919ff4be931da9c62173959971f9944bfc6d304ee80ab09448d79f6b336` |
| battery_try | `5ebaa98a5bf1151f9480811173cf5cd4de0a5e53ca6e123dea75a997bbb0be89` |
| cover_blocks | `ca4cf5ffdf648b61bcfa63e40532ab285b1368ceff728efa53fafa60bd27e551` |

converter 配置位于 `converter_configs/state_token_serial_soft/`。phase 在推理时只允许保持或
前进一步；attribute 的 0 类是 unknown，可解析为任一合法值，首次变为非零后锁存。
旧 `rearrange_blocks` 的按钮状态特殊约束保持向后兼容，但本实验四个任务都不使用按钮 guard。

数据转换器把 resolved config 和源数据 provenance 写入每个 LeRobot repo 的
`meta/rmbench/`。训练入口再把这四个文件复制到 checkpoint 的
`metadata/rmbench_data_meta/`，因此后续评测从 checkpoint 恢复每个任务的实际 schema，
不依赖共享 train config 的默认 task，也不通过 `policy_metadata` 传实验信息。

## 运行入口

正式训练由一个 manifest 定义，一张卡一个任务：

```bash
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_state_token_serial_soft/jobs_train.json \
  --gpus 4,5,6,7 \
  --state policy/pi05/checkpoints/pi05_multitask_state_token_serial_soft/train_queue_state.json
```

目标 checkpoint：

```text
policy/pi05/checkpoints/pi05_multitask_state_token_serial_soft/<task>_seed42/<step>
```

正式训练于 2026-08-10 01:44 CST 从 clean commit `e062e2e` 启动：

| Run | GPU | W&B ID | 启动状态 |
| --- | ---: | --- | --- |
| put_back_block_seed42 | 4 | `dduk3wcd` | completed |
| swap_blocks_seed42 | 5 | `ullelo12` | completed |
| battery_try_seed42 | 6 | `8bi475al` | completed |
| cover_blocks_seed42 | 7 | `4y08q2u8` | completed |

训练队列于 2026-08-11 07:50 CST 完成，4/4 succeeded、0 failed，四个 run 均保存
10k/20k/30k checkpoint。

## 正式评测

四个任务均评测 30k checkpoint、seed0、100 rollouts，并保存5个带 state-token overlay 的
视频。为比较 replanning 间隔，同时运行 `pi0_step=30/50`：

```bash
# 本机 GPU3-6
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_state_token_serial_soft/jobs_eval.json \
  --gpus 3,4,5,6 \
  --state eval_result/pi05_multitask_state_token_serial_soft/eval_step30_queue_state.json

# wuwen-12 GPU2-5
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_state_token_serial_soft/jobs_eval_step50.json \
  --gpus 2,3,4,5 \
  --state eval_result/pi05_multitask_state_token_serial_soft/eval_step50_queue_state.json
```

结果目录为：

```text
eval_result/pi05_multitask_state_token_serial_soft/<task>_seed42@ckpt30k_step<30|50>_100ep_seed<0|1>
```

两组评测于 2026-08-11 13:17 CST 启动，模型均已加载成功：

| Task | step30（本机 GPU / W&B） | step50（wuwen-12 GPU / W&B） |
| --- | --- | --- |
| put_back_block | GPU3 / `8ejh2fhu` | GPU2 / `mmezmhag` |
| swap_blocks | GPU4 / `hf0ynaqf` | GPU3 / `ue9abnan` |
| battery_try | GPU5 / `i30p6cgp` | GPU4 / `69ow0226` |
| cover_blocks | GPU6 / `yfbawiwh` | GPU5 / `1bh0k6dr` |

seed1 复现继续使用相同 checkpoint 和评测设置。启动时 seed0 的6个长任务仍在运行，实时空闲卡为
本机 GPU0/3/7 与 wuwen-12 GPU0/1/2/6/7，因此将8项拆为3+5并行：

```bash
# 本机 GPU0/3/7：step30 的 put_back_block / swap_blocks / battery_try
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_state_token_serial_soft/jobs_eval_seed1_local.json \
  --gpus 0,3,7 \
  --state eval_result/pi05_multitask_state_token_serial_soft/eval_seed1_local_queue_state.json

# wuwen-12 GPU0/1/2/6/7：step30 cover_blocks + 四个 step50
python script/run_job_queue.py \
  --jobs experiments/pi05_multitask_state_token_serial_soft/jobs_eval_seed1_wuwen12.json \
  --gpus 0,1,2,6,7 \
  --state eval_result/pi05_multitask_state_token_serial_soft/eval_seed1_wuwen12_queue_state.json
```

seed1 于 2026-08-11 16:03 CST 启动，8个模型均已加载成功：

| Task | step30（机器 / GPU / W&B） | step50（机器 / GPU / W&B） |
| --- | --- | --- |
| put_back_block | 本机 / GPU0 / `o9egunjc` | wuwen-12 / GPU1 / `pj4erxv2` |
| swap_blocks | 本机 / GPU3 / `ggq7ox46` | wuwen-12 / GPU2 / `2q6e82bu` |
| battery_try | 本机 / GPU7 / `rik89hiy` | wuwen-12 / GPU6 / `2x01vgde` |
| cover_blocks | wuwen-12 / GPU0 / `73lgv9bx` | wuwen-12 / GPU7 / `higjuz5i` |

## 验证与状态

- Python lint：completed
- converter/config/policy regression：completed（13 passed）
- model state-token regression：completed（12 passed）
- 200 episode metadata/state-label resolution audit：completed
- 1-episode LeRobot conversion smoke：completed（四任务）
- real data batch：completed（bs=32，字段数 2 / 3 / 1 / 4）
- full LeRobot conversion：completed（四任务各 50 episodes）
- 10k-frame norm stats：completed（四任务，实际各 9,984 frames）
- 2-step train smoke：completed（4/4 return code 0）
- formal 30k training：completed（4/4 succeeded，0 failed）
- formal step30 evaluation：running（本机 GPU3–6；四任务，seed0，各100 episodes）
- formal step50 evaluation：running（wuwen-12 GPU2–5；四任务，seed0，各100 episodes）
- eval seed1 replication：running（8/8 已加载；step30/50 × 四任务，各100 episodes）
