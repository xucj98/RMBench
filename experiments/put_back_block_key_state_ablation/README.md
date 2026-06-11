# Put Back Block Key-State 消融

Batch ID: `put_back_block_key_state_ablation`

本目录记录 `put_back_block` 上第一批 key-state pi0 LoRA 探索实验。当前结论是：这批实验的训练和数据生成过程还没有完成审计，已评测结果也不好，因此不要把它当作可复现的正式实验批次使用。这里先保留已经能确认的事实，方便排查 failure case 和后续重新设计实验。

## 当前状态

```text
task: put_back_block
policy: pi05 deploy pi0 checkpoint
checkpoint_id: 30000
已确认 checkpoint: 8 个 variant 在共享存储上存在
当前保留 eval: default variant 在修复 key-state 推理输入后完成 100-rollout raw/schema_latch 对照，二者均为 55/100
最近排查: schema_latch 不提升当前 default 模型成功率；失败主要卡在按钮 press gate，见下方排查记录
训练可复现性: 暂未确认
```

训练复跑暂时不要按本文档执行。历史 launcher 和训练配置还需要结合 git 历史、数据转换脚本和 wandb metadata 重新核对。

## 变体定义

8 个 variant 的定义来自 `policy/pi05/scripts/run_put_back_block_key_state_experiments.py` 和 `policy/pi05/src/openpi/training/config.py`。下表只说明当前代码里能看到的语义，不表示这些训练过程已经可复现。

| Variant | Train config | Model name | Dataset repo id | Key-state 设置 |
| --- | --- | --- | --- | --- |
| `default` | `pi0_aloha_put_back_block_key_state_default_lora` | `pi0_put_back_block_key_state_default` | `put_back_block_demo_clean_key_state_default` | phase 使用 GT；mat 在 `W_mat` 结束前保持 unknown；无 margin；逐步输出 key-state |
| `mat_first` | `pi0_aloha_put_back_block_key_state_mat_first_lora` | `pi0_put_back_block_key_state_mat_first` | `put_back_block_demo_clean_key_state_mat_first` | phase 使用 GT；mat 只在第一帧 unknown；逐步输出 key-state |
| `mat_hash_p50` | `pi0_aloha_put_back_block_key_state_mat_hash_p50_lora` | `pi0_put_back_block_key_state_mat_hash_p50` | `put_back_block_demo_clean_key_state_mat_hash_p50` | phase 使用 GT；早期 mat hash mix；`mat_unknown_prob=0.5` |
| `wmat_margin10` | `pi0_aloha_put_back_block_key_state_wmat_margin10_lora` | `pi0_put_back_block_key_state_wmat_margin10` | `put_back_block_demo_clean_key_state_wmat_margin10` | mat 在 `W_mat` 结束后再延迟 10 帧变为 known |
| `wmat_margin20` | `pi0_aloha_put_back_block_key_state_wmat_margin20_lora` | `pi0_put_back_block_key_state_wmat_margin20` | `put_back_block_demo_clean_key_state_wmat_margin20` | mat 在 `W_mat` 结束后再延迟 20 帧变为 known |
| `phase_lag10` | `pi0_aloha_put_back_block_key_state_phase_lag10_lora` | `pi0_put_back_block_key_state_phase_lag10` | `put_back_block_demo_clean_key_state_phase_lag10` | phase 在 boundary 后滞后 10 帧；启用 lag recovery |
| `phase_lag20` | `pi0_aloha_put_back_block_key_state_phase_lag20_lora` | `pi0_put_back_block_key_state_phase_lag20` | `put_back_block_demo_clean_key_state_phase_lag20` | phase 在 boundary 后滞后 20 帧；启用 lag recovery |
| `phase_jitter5` | `pi0_aloha_put_back_block_key_state_phase_jitter5_lora` | `pi0_put_back_block_key_state_phase_jitter5` | `put_back_block_demo_clean_key_state_phase_jitter5` | phase boundary 加 5 帧 jitter |

当前代码路径里，这 8 个 variant 都使用 `key_output_mode=per_step`。

## 检查点

以下 step-30000 checkpoint 目录已经确认存在：

```text
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_default_lora/pi0_put_back_block_key_state_default/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_mat_first_lora/pi0_put_back_block_key_state_mat_first/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_mat_hash_p50_lora/pi0_put_back_block_key_state_mat_hash_p50/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_wmat_margin10_lora/pi0_put_back_block_key_state_wmat_margin10/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_wmat_margin20_lora/pi0_put_back_block_key_state_wmat_margin20/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_phase_lag10_lora/pi0_put_back_block_key_state_phase_lag10/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_phase_lag20_lora/pi0_put_back_block_key_state_phase_lag20/30000
/mnt/public3/xcj/rmbench/pi0_checkpoints/pi0_aloha_put_back_block_key_state_phase_jitter5_lora/pi0_put_back_block_key_state_phase_jitter5/30000
```

`policy/pi05/pi_model.py` 当前从下面的路径加载 eval checkpoint：

```text
policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>
```

写本文档时，只有下面 3 个 key-state variant 能通过 `policy/pi05/checkpoints` 访问，并且已经做过历史 eval：

```text
default
mat_hash_p50
phase_jitter5
```

如果要评测其他 variant，先确认对应 checkpoint 能在 `policy/pi05/checkpoints/<train_config_name>/<model_name>/30000` 访问。

## 训练

训练来源暂时还不可靠，不应写成可复现入口。

已知相关代码位置：

```text
policy/pi05/scripts/run_put_back_block_key_state_experiments.py
policy/pi05/src/openpi/training/config.py
policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py
```

写训练复现命令前，应先检查下面几个历史 commit 中的数据生成和转换逻辑：

```text
80c224c2c58c7ae24ff52bb553471ac5a64a59d6
f6de7ff05ed5311a535b1006093825d9b1b3963e
1eeadd24eaf9e533ed45ceb6b22e1f7f390e9f45
```

在数据源、转换命令、norm stats 生成、训练命令、运行 commit 和 wandb metadata 对齐之前，不要把旧 launcher 当成权威训练入口。

## 评测

当前 repo 级 eval 入口是：

```text
script/eval_policy.py
```

pi05 eval 使用 pi05 virtualenv 直接运行：

```bash
policy/pi05/.venv/bin/python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name put_back_block \
  --task_config demo_clean_eval \
  --train_config_name <train_config_name> \
  --model_name <model_name> \
  --ckpt_setting <ckpt_setting> \
  --seed 0 \
  --policy_name pi05 \
  --test_num 100 \
  --eval_video_log true \
  --eval_video_count 5 \
  --eval_video_key_state_overlay true
```

注意：`script/eval_policy.py` 默认 `test_num=100`。`task_config/demo_clean_eval.yml` 里的 `episode_num: 50` 不会控制 eval rollout 数。RMBench 论文中的模拟实验表格也是 100 rollouts；之前本地 key-state 和 pi0 baseline 记录里用过 50 rollouts，那是探索设置，不是论文标准设置。

录制 eval 视频时，`envs/_base_task.py::get_obs_for_policy()` 会退回完整 `get_obs()`，不会走 `get_obs_fast()`。这对 failure-case 视频是必要的；不录视频的 episode 仍可能使用 `get_obs_fast()`。

后续 key-state 相关视频评测默认开启 `eval_video_key_state_overlay`。overlay 中显示的 key-state 用来排查策略输出和任务阶段，不代表环境成功条件本身。

## 当前保留结果

下面 3 个 overlay eval 是当前保留的 layout/failure-case 记录。它们评测的是同一批 key-state checkpoint，因此保留在本实验批次下，不单独建目录。每个目录下有 `episode0.mp4` 到 `episode4.mp4`，视频左上角带 key-state overlay。三条结果成功率均为 0.0。

| Variant | Rollouts | Video count | Success rate | Source |
| --- | ---: | ---: | ---: | --- |
| `default` | 5 | 5 | 0.0 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_ks_overlay_video5/2026-06-09 01:31:21/_result.txt` |
| `mat_hash_p50` | 5 | 5 | 0.0 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_mat_hash_p50_mem_ks_overlay_video5/2026-06-09 01:31:22/_result.txt` |
| `phase_jitter5` | 5 | 5 | 0.0 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_phase_jitter5_mem_ks_overlay_video5/2026-06-09 01:31:22/_result.txt` |

## 2026-06-10 锁存视频排查

为排查 overlay 中 key-state 中途跳变的问题，代码增加了可选的 `key_state_update_mode=schema_latch`。该模式不改变默认评测语义；只有显式指定时才会使用 schema 里的 update rule，让 phase 单步单调推进、mat 首次识别后锁存。

下面这次运行是调试性视频排查，启动后在看到 failure case 后手动停止，不作为正式 eval 指标：

```text
ckpt_setting: pi0_put_back_block_key_state_default_mem_schema_latch_video5
train_config_name: pi0_aloha_put_back_block_key_state_default_lora
model_name: pi0_put_back_block_key_state_default
checkpoint_id: 30000
key_state_update_mode: schema_latch
test_num: 5
eval_video_log: true
eval_video_count: 5
eval_video_key_state_overlay: true
result_dir: eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_schema_latch_video5/2026-06-10 21:54:17
```

已生成文件：

```text
episode0.mp4: 完整生成，eval_log 记录 Fail
episode1.mp4: 手动停止时的中间产物，不作为完整 episode
_result.txt: 未生成
```

初步观察：即使使用 `schema_latch` 稳定 overlay 中的 phase 和 mat，视频里策略看起来仍然没有根据 phase/mat state 切换到不同运动模式；它仍主要在执行 phase 0 的动作，即把物体放到正中间，而不是在后续阶段把物体放回对应 mat。后续检查发现，部署推理时已经生成了 32 维 key-state policy state，但实际传给 policy 的仍是原始 14 维 state。该问题已在 `a6737e5` 修复。

## 2026-06-10 State Input 修复后对照

在 `a6737e5` 修复 pi05 部署时 key-state 推理输入丢失的问题后，对 `default` variant 重新做了 100-rollout eval。两次评测均开启前 5 个 episode 的 overlay 视频录制。

| Run | key_state_update_mode | Rollouts | Success | Success rate | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `statefix_raw_100rollout_video5` | `raw` | 100 | 55 | 0.55 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_raw_100rollout_video5/2026-06-10 22:41:07/_result.txt` |
| `statefix_schema_latch_100rollout_video5` | `schema_latch` | 100 | 55 | 0.55 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_schema_latch_100rollout_video5/2026-06-10 22:41:07/_result.txt` |

两组 eval 的 episode 结果逐条一致：`seed=100000` 到 `seed=100099` 中均为 55 个 Success、45 个 Fail。因此，当前 default 模型的成功率不依赖 `schema_latch`；正常 eval 可以继续使用默认 `raw`，`schema_latch` 主要保留为 overlay/阶段状态排查工具。

为解释失败原因，额外做过一次临时 debug 统计：`schema_latch`、20 rollouts、无视频，结果为 11/20。9 个失败 episode 全部满足：

```text
press_cnt = 0
stage_id 从未离开 0
block 曾到达 center
block 曾满足最终目标区域几何条件
```

这说明这批失败不是最终放置位置本身没到，而是环境没有记录到按钮 press，导致 `stage_id` 没有打开最终成功 gate。相关 debug 代码只用于本地排查，不作为正式实验代码提交。

## 2026-06-11 Full Finetune 对照

为比较同一份 default key-state 数据下 LoRA 和全量微调的差异，启动一个 pi0 full finetune 对照。当前 GPU7 空闲，本次使用单卡 `batch_size=32`，并通过 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 提高 XLA 可预分配显存上限。

```text
train_config_name: pi0_aloha_put_back_block_key_state_default_full_b32
model_name: pi0_put_back_block_key_state_default_full_b32
dataset repo id: put_back_block_demo_clean_key_state_default
fine_tune: full
batch_size: 32
num_train_steps: 30000
gpu: 7
checkpoint_dir: policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_full_b32/pi0_put_back_block_key_state_default_full_b32
```

该 run 是 full finetune follow-up，不计入上面 8 个 LoRA 消融 variant。若单卡 full finetune 仍出现 OOM，再改为等待多卡空闲后按 FSDP 复跑。

## 历史旁证

下面是 3 个已评测 variant 的历史 50-rollout 结果：

| Variant | Result | Source |
| --- | ---: | --- |
| `default` | 0/50 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem/2026-06-03 15:29:15/_result.txt` |
| `mat_hash_p50` | 0/50 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_mat_hash_p50_mem/2026-06-03 15:29:15/_result.txt` |
| `phase_jitter5` | 0/50 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_phase_jitter5_mem/2026-06-03 15:29:14/_result.txt` |

用于对照的 pi0 baseline：

```text
pi0_lora_baseline put_back_block: 4/50 = 8%
source: eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block/2026-06-01 16:27:39/_result.txt
```

2026-06-09 的非 overlay 100-rollout rerun 也存在：`default` 和 `mat_hash_p50` 为 0/100，`phase_jitter5` 为 1/100。这组结果没有 key-state overlay，不作为当前 failure-case 视频记录；后续分析优先使用上面的 overlay 目录。

名字里有 `video5_rerun` 但没有 `100rollout` 的中断目录，以及名字里有 `50rollout_video5_rerun` 的中断目录，都不要作为结果使用。

## 新实验前需要补齐

重新启动 key-state 实验批次前，应写一个新的实验 README 和 launcher，并明确记录：

```text
batch_id
commit
data source and conversion command
train command
checkpoint path
eval command
wandb project/group/id
success_count / test_num / success_rate
```

wandb 使用 `project=RMBench`、`group=<batch_id>`。smoke test、中断运行和未完成 eval 不进入结果表。
