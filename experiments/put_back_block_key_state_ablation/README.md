# Put Back Block Key-State 消融

Batch ID: `put_back_block_key_state_ablation`

本目录记录 `put_back_block` 上第一批 key-state pi0 LoRA 探索实验。当前结论是：这批实验的训练和数据生成过程还没有完成审计，已评测结果也不好，因此不要把它当作可复现的正式实验批次使用。这里先保留已经能确认的事实，方便排查 failure case 和后续重新设计实验。

## 当前状态

```text
task: put_back_block
policy: pi05 deploy pi0 checkpoint
checkpoint_id: 30000
已确认 checkpoint: 8 个 variant 在共享存储上存在
历史 eval: 3 个 variant，各 50 rollouts，均为 0/50
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
  --eval_video_count 5
```

注意：`script/eval_policy.py` 默认 `test_num=100`。`task_config/demo_clean_eval.yml` 里的 `episode_num: 50` 不会控制 eval rollout 数。RMBench 论文中的模拟实验表格也是 100 rollouts；之前本地 key-state 和 pi0 baseline 记录里用过 50 rollouts，那是探索设置，不是论文标准设置。

录制 eval 视频时，`envs/_base_task.py::get_obs_for_policy()` 会退回完整 `get_obs()`，不会走 `get_obs_fast()`。这对 failure-case 视频是必要的；不录视频的 episode 仍可能使用 `get_obs_fast()`。

## 历史结果

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

2026-06-09 启动的 failure-video rerun 使用 100 rollouts，并录制前 5 个 episode：

| Variant | ckpt_setting | GPU | Launch log |
| --- | --- | ---: | --- |
| `default` | `pi0_put_back_block_key_state_default_mem_100rollout_video5_rerun` | 2 | `eval_result/put_back_block/pi05/demo_clean_eval/_launch_logs/pi0_put_back_block_key_state_default_mem_100rollout_video5_rerun_gpu2_20260609_011500.log` |
| `mat_hash_p50` | `pi0_put_back_block_key_state_mat_hash_p50_mem_100rollout_video5_rerun` | 3 | `eval_result/put_back_block/pi05/demo_clean_eval/_launch_logs/pi0_put_back_block_key_state_mat_hash_p50_mem_100rollout_video5_rerun_gpu3_20260609_011500.log` |
| `phase_jitter5` | `pi0_put_back_block_key_state_phase_jitter5_mem_100rollout_video5_rerun` | 4 | `eval_result/put_back_block/pi05/demo_clean_eval/_launch_logs/pi0_put_back_block_key_state_phase_jitter5_mem_100rollout_video5_rerun_gpu4_20260609_011501.log` |

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
