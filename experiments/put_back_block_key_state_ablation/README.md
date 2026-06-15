# Put Back Block Key-State 消融

Batch ID: `put_back_block_key_state_ablation`

本实验批次围绕 `put_back_block` 的 key-state memory 方案。核心问题是：pi0 单帧观察很难在后期恢复 block 原始所在 mat，因此我们给 policy 增加外部 key-state memory 输入，并让 policy 在 action 中预测下一步 memory。

## 结论

1. key-state memory 明显提升 `put_back_block`：pi0 LoRA baseline 正式 100-rollout 结果是 7/100 = 0.07，早期 50-rollout 结果是 4/50 = 0.08；key-state default LoRA 是 27/50 = 0.54，另一次 100-rollout 复测是 55/100 = 0.55。
2. 本批实验只验证 Scheme B，即 per-frame/per-step key-state target；没有实现或评测 Scheme A 的 chunk-level repeated key-state target。
3. 在 Scheme B 下，`default` 数据处理最好。`mat_first`、`mat_hash_p50`、`wmat_margin10/20`、`phase_lag10/20`、`phase_jitter5` 都没有超过 default。
4. 当前最合理的解释是：仿真数据干净、数据量小，额外的鲁棒性设计反而引入 label noise、监督稀释或状态和动作不一致。
5. default full finetune 达到 68/100 = 0.68，比 default LoRA 的 55/100 = 0.55 更高，但仍低于论文报告的 Mem-0 90%，按钮阶段失败仍需要继续排查。

## 实验问题

本批实验回答两个问题：

1. 在 `put_back_block` 上，外部 key-state memory 是否能显著优于普通 pi0 LoRA baseline。
2. 在已选定 per-step key-state 的 Scheme B 后，哪些 key-state 数据处理细节值得保留为默认方案。

不回答的问题：

1. 不比较 Scheme A 和 Scheme B。Scheme A 需要自定义 dataset/transform 或单独监督头，本批没有实现。
2. 不证明这些鲁棒性设计在真实噪声数据上无效。本批数据来自仿真，且只有 50 条 demo。
3. 不把旧训练 launcher 当成完整可复现入口。旧 LoRA 训练的数据生成和启动过程还没有完成审计。

设计细节见 [docs/put_back_block_key_state_design.md](../../docs/put_back_block_key_state_design.md)。

## 方法摘要

key-state 包含两个离散状态：

```text
phase_id:
  0 move_block_to_center
  1 press_button
  2 move_block_back_to_origin_mat

mat_id:
  0 unknown
  1 left
  2 right
  3 front
  4 back
```

pi0 的 `observation.state` 和 `action` 从 14 维机器人状态扩到 32 维：

```text
dim 0:14   robot qpos / robot action
dim 14:17  phase one-hot
dim 17:22  mat one-hot
dim 22:32  zero padding
```

本批使用 Scheme B：

```text
action[f, key_dims] = key state after action f
```

因此 LeRobot 按连续 frame 拼出的 action chunk 中，key-state target 是一条 per-step trajectory。推理时执行 `pi0_step` 个 robot action 后，用第 `pi0_step - 1` 个 action 中的 key-state 预测更新下一次 query 的 memory。

## Variant 定义

8 个 LoRA variant 都使用 `key_output_mode=per_step`，差别只在 key-state input/label 构造。

| Variant | 设计意图 | 数据处理差异 |
| --- | --- | --- |
| `default` | 主方案 | phase 输入使用 GT；mat 在 segment 0 内为 unknown，segment 0 结束后变为原始 mat；无 margin、无 jitter |
| `mat_first` | 测试更早给出 mat memory 是否更好 | 只有第一帧 mat 为 unknown，之后全部给原始 mat |
| `mat_hash_p50` | 测试 early window 内 known/unknown 混合是否更鲁棒 | segment 0 内用固定 hash 让 50% frame 为 unknown，其余给原始 mat |
| `wmat_margin10` | 测试更宽 mat acquisition window | segment 0 结束后再延迟 10 帧才给原始 mat |
| `wmat_margin20` | 测试更宽 mat acquisition window | segment 0 结束后再延迟 20 帧才给原始 mat |
| `phase_lag10` | 测试 phase memory 滞后时的 recovery | phase boundary 后 10 帧输入仍保持旧 phase，target 使用真实 phase |
| `phase_lag20` | 测试更强 phase lag recovery | phase boundary 后 20 帧输入仍保持旧 phase，target 使用真实 phase |
| `phase_jitter5` | 测试 phase 边界不确定性 | 每条 episode 的 phase boundary 加 `[-5, 5]` 帧 jitter |

训练配置和数据集：

| Variant | Train config | Model name | Dataset repo id |
| --- | --- | --- | --- |
| `default` | `pi0_aloha_put_back_block_key_state_default_lora` | `pi0_put_back_block_key_state_default` | `put_back_block_demo_clean_key_state_default` |
| `mat_first` | `pi0_aloha_put_back_block_key_state_mat_first_lora` | `pi0_put_back_block_key_state_mat_first` | `put_back_block_demo_clean_key_state_mat_first` |
| `mat_hash_p50` | `pi0_aloha_put_back_block_key_state_mat_hash_p50_lora` | `pi0_put_back_block_key_state_mat_hash_p50` | `put_back_block_demo_clean_key_state_mat_hash_p50` |
| `wmat_margin10` | `pi0_aloha_put_back_block_key_state_wmat_margin10_lora` | `pi0_put_back_block_key_state_wmat_margin10` | `put_back_block_demo_clean_key_state_wmat_margin10` |
| `wmat_margin20` | `pi0_aloha_put_back_block_key_state_wmat_margin20_lora` | `pi0_put_back_block_key_state_wmat_margin20` | `put_back_block_demo_clean_key_state_wmat_margin20` |
| `phase_lag10` | `pi0_aloha_put_back_block_key_state_phase_lag10_lora` | `pi0_put_back_block_key_state_phase_lag10` | `put_back_block_demo_clean_key_state_phase_lag10` |
| `phase_lag20` | `pi0_aloha_put_back_block_key_state_phase_lag20_lora` | `pi0_put_back_block_key_state_phase_lag20` | `put_back_block_demo_clean_key_state_phase_lag20` |
| `phase_jitter5` | `pi0_aloha_put_back_block_key_state_phase_jitter5_lora` | `pi0_put_back_block_key_state_phase_jitter5` | `put_back_block_demo_clean_key_state_phase_jitter5` |

## 主实验

评测设置：

```text
eval commit: d9eb4d0ef3e8a3cbf37242d30b1d0c35cb3bfd9a
task: put_back_block
policy: pi05 deploy pi0 checkpoint
task_config: demo_clean_eval
checkpoint_id: 30000
test_num: 50
seed: 0
eval seeds: 100000..100049
eval_video_log: true
eval_video_count: 2
eval_video_key_state_overlay: true
key_state_update_mode: raw
queue_log: eval_result/put_back_block_key_state_ablation/_queue_20260611_214744.log
```

每个结果目录下有 `_result.txt`、`eval_log.txt`、`stdout.log`、`config.yaml` 和 `command.txt`。`config.yaml` 保存合并后的配置快照，`command.txt` 由 eval 代码直接记录启动命令和 git commit。

结果：

| Variant | Success | Success rate | Result dir |
| --- | ---: | ---: | --- |
| pi0 baseline | 4/50 | 0.08 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block/2026-06-01 16:27:39` |
| `default` | 27/50 | 0.54 | `eval_result/put_back_block_key_state_ablation/default_raw_50_video2_20260611_214744` |
| `mat_hash_p50` | 21/50 | 0.42 | `eval_result/put_back_block_key_state_ablation/mat_hash_p50_raw_50_video2_20260611_214744` |
| `wmat_margin10` | 19/50 | 0.38 | `eval_result/put_back_block_key_state_ablation/wmat_margin10_raw_50_video2_20260611_214744` |
| `phase_jitter5` | 19/50 | 0.38 | `eval_result/put_back_block_key_state_ablation/phase_jitter5_raw_50_video2_20260611_214744` |
| `phase_lag10` | 18/50 | 0.36 | `eval_result/put_back_block_key_state_ablation/phase_lag10_raw_50_video2_20260611_214744` |
| `mat_first` | 16/50 | 0.32 | `eval_result/put_back_block_key_state_ablation/mat_first_raw_50_video2_20260611_214744` |
| `wmat_margin20` | 15/50 | 0.30 | `eval_result/put_back_block_key_state_ablation/wmat_margin20_raw_50_video2_20260611_214744` |
| `phase_lag20` | 9/50 | 0.18 | `eval_result/put_back_block_key_state_ablation/phase_lag20_raw_50_video2_20260611_214744` |

default 另有一次 100-rollout 复测：

| Run | key_state_update_mode | Success | Success rate | Result |
| --- | --- | ---: | ---: | --- |
| `statefix_raw_100rollout_video5` | `raw` | 55/100 | 0.55 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_raw_100rollout_video5/2026-06-10 22:41:07` |
| `statefix_schema_latch_100rollout_video5` | `schema_latch` | 55/100 | 0.55 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_schema_latch_100rollout_video5/2026-06-10 22:41:07` |

两组 100-rollout default eval 的 episode 结果逐条一致，因此当前 default 模型不依赖 `schema_latch`；主结果采用默认 `raw`。

## 结果解释

`default` 是最好的数据处理方案。它给模型足够多的早期 `mat=unknown` 样本，让模型学习从早期图像中预测 origin mat；segment 0 结束后再把 mat 写入 memory，避免在后期不可观测图像中重新估计 origin mat。

`mat_first` 显著变差，说明只在每条 episode 第一帧监督 unknown -> mat 不够。`mat_hash_p50` 是第二好，但仍低于 default；它减少了 unknown acquisition 监督，同时让 early memory 状态更不一致。

`wmat_margin10/20` 没有收益，且 margin 越大越差。文档里的默认设计认为 segment 0 是最可靠的 mat acquisition window；进入 segment 1 后 block 已经被抬起，origin mat 的直接视觉证据变弱。

`phase_lag10/20` 和 `phase_jitter5` 都低于 default。它们本来是鲁棒性设计，但当前实现是在原始 frame 上替换 phase input 或扰动 boundary，不复制 recovery 样本，也不改 sampler。对于干净仿真数据和小数据量，这更像引入状态和动作不一致或 label noise。

因此，本批实验支持的默认方案是：Scheme B per-step key-state target、GT phase input、`unknown_until_wmat_end` mat input、`wmat_margin_frames=0`、无 phase jitter、无 phase lag。

## Follow-Up: Default Full Finetune

动机：default LoRA 约 55% 成功率，失败主要集中在按钮阶段。为判断瓶颈是否来自 LoRA 容量或训练方式，补充了同一份 default key-state 数据上的 full finetune。

训练和评测状态：

```text
status: trained, eval complete
train commit: de1de697d1ab80478afb3852cc9d7d64102aa112
wandb project: RMBench
wandb id: u3csuoca
train_config_name: pi0_aloha_put_back_block_key_state_default_full_b32
model_name: pi0_put_back_block_key_state_default_full_b32
dataset repo id: put_back_block_demo_clean_key_state_default
fine_tune: full
batch_size: 32
num_train_steps: 30000
checkpoint_id: 30000
checkpoint_dir: policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_full_b32/pi0_put_back_block_key_state_default_full_b32/30000
eval_result_dir: eval_result/put_back_block_key_state_ablation/default_full_b32_raw_100_video5_20260614_170220
eval_queue_log: eval_result/put_back_block_key_state_ablation/_eval_default_full_b32_raw_100_video5_20260614_170220.log
eval_success: 68/100
eval_success_rate: 0.68
```

训练命令来自本地 wandb metadata：

```bash
cd policy/pi05
.venv/bin/python scripts/train.py \
  pi0_aloha_put_back_block_key_state_default_full_b32 \
  --exp-name=pi0_put_back_block_key_state_default_full_b32 \
  --checkpoint-base-dir=checkpoints
```

该 follow-up 属于同一 batch，但不计入 8 个数据处理消融 variant。

对照结果：

| Method | Setting | Success | Success rate | Source |
| --- | --- | ---: | ---: | --- |
| pi0 LoRA baseline | 原始 observation，无 key-state | 7/100 | 0.07 | `eval_result/pi0_lora_baseline/put_back_block_raw_100_video5_20260614_pi0_baseline_100` |
| Paper Pi0.5 | RMBench Table 1 | - | 0.11 | `PROGRESS.md` 论文基准表 |
| Paper Mem-0 | RMBench Table 1 | - | 0.90 | `PROGRESS.md` 论文基准表 |
| key-state default LoRA | Scheme B, raw memory update | 55/100 | 0.55 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_raw_100rollout_video5/2026-06-10 22:41:07` |
| key-state default full finetune | Scheme B, full finetune, bs=32 | 68/100 | 0.68 | `eval_result/put_back_block_key_state_ablation/default_full_b32_raw_100_video5_20260614_170220` |

相对 pi0 LoRA baseline，key-state default LoRA 提升 48 pp，full finetune 提升 61 pp；相对论文 Pi0.5 也明显更高。full finetune 相比 LoRA default 提升 13 pp，说明容量或训练方式确实是瓶颈之一；但 0.68 仍低于论文报告的 Mem-0 0.90，按钮阶段失败仍需要继续排查。

## 产物

LoRA checkpoint 根目录：

```text
policy/pi05/checkpoints/<train_config_name>/<model_name>/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/<train_config_name>/<model_name>/30000
```

full finetune checkpoint：

```text
policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_full_b32/pi0_put_back_block_key_state_default_full_b32/30000
/mnt/public3/xcj/rmbench/pi05_checkpoints/pi0_aloha_put_back_block_key_state_default_full_b32/pi0_put_back_block_key_state_default_full_b32/30000
```

主 eval 根目录：

```text
eval_result/put_back_block_key_state_ablation
```

训练和数据生成的完整可复现性仍未审计。旧 launcher、数据转换命令和 LoRA 训练 wandb metadata 需要后续从 git 历史和本地产物中补齐；不要把旧 LoRA 训练过程当成已经完整归档的正式入口。

## 附录

调试记录见 [notes_debug.md](notes_debug.md)。早期不作为正式指标的历史结果见 [legacy_results.md](legacy_results.md)。
