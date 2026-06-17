# RMBench DP Table 1 复现

Batch ID: `dp_table1_reproduction`

本批实验记录 RMBench Table 1 中 DP policy 在 9 个任务上的复现结果。目标是复核 50 demos、single-task DP、100 rollout eval 下的成功率，并和论文 DP 数值对齐。

`experiments/` 只记录批次说明和启动入口。具体实验事实以 checkpoint 目录、`eval_result` 目录和 wandb 为准；本批历史结果还没有统一 metadata，因此不额外维护 `runs.yaml`。

## 实验范围

正式结果范围：

```text
policy: DP
train data: policy/DP/data/<task>-demo_clean-50.zarr
checkpoint: policy/DP/checkpoints/<task>-demo_clean-50-0/600.ckpt
eval config: policy/DP/deploy_policy.yml
task config: demo_clean
instruction_type: unseen
checkpoint_num: 600
expert_data_num: 50
seed: 0
test_num: 100
```

不纳入正式结果：

```text
smoke test
启动测试
2 rollout / 5 rollout / 20 rollout 调试评测
没有 _result.txt 的未完成评测目录
```

## 运行方式

训练单个任务：

```bash
cd policy/DP
bash train.sh <task> demo_clean 50 0 14 <gpu_id>
```

评测单个任务：

```bash
cd policy/DP
export LD_LIBRARY_PATH="/root/miniconda3/envs/RMBench/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}"
bash eval.sh <task> demo_clean default 50 0 <gpu_id>
```

批量评测 9 个 DP checkpoint：

```bash
bash helper/auto_eval_dp.sh
```

`helper/auto_eval_dp.sh` 当前写死了 repo 路径和 `CUDA_VISIBLE_DEVICES=2`。换机器或换 GPU 时先检查脚本。

## 产物规则

训练产物：

```text
policy/DP/checkpoints/<task>-demo_clean-50-0/
```

评测产物：

```text
eval_result/dp_reproduction/<task>/
```

wandb：

```text
project: RMBench
expected group for future reruns: dp_table1_reproduction
historical DP run name pattern: DP_<task>
```

历史 DP 训练曾在 wandb 启用前后各有运行，当前 README 不硬补缺失的 wandb id。后续正式复跑应在 checkpoint metadata、eval_result metadata 或 wandb config 中记录 `batch_id=dp_table1_reproduction`、run name、commit、训练命令、评测命令、checkpoint 引用和 eval_result 引用。

## 结果

下表只采用 100-rollout 正式评测。`swap_T` 和 `battery_try` 的 20-rollout 调试结果不计入本表。

| Task | Eval result | Success | Paper DP | Source |
| --- | ---: | ---: | ---: | --- |
| `observe_and_pickup` | 2/100 | 2% | 1% | `eval_result/dp_reproduction/observe_and_pickup/_result.txt` |
| `put_back_block` | 0/100 | 0% | 0% | `eval_result/dp_reproduction/put_back_block/_result.txt` |
| `rearrange_blocks` | 0/100 | 0% | 0% | `eval_result/dp_reproduction/rearrange_blocks/_result.txt` |
| `swap_T` | 11/100 | 11% | 20% | `eval_result/dp_reproduction/swap_T/_result.txt` |
| `swap_blocks` | 15/100 | 15% | 11% | `eval_result/dp_reproduction/swap_blocks/_result.txt` |
| `cover_blocks` | 0/100 | 0% | 0% | `eval_result/dp_reproduction/cover_blocks/_result.txt` |
| `battery_try` | 13/100 | 13% | 10% | `eval_result/dp_reproduction/battery_try/_result.txt` |
| `press_button` | 0/100 | 0% | 0% | `eval_result/dp_reproduction/press_button/_result.txt` |
| `blocks_ranking_try` | 3/100 | 3% | 10% | `eval_result/dp_reproduction/blocks_ranking_try/_result.txt` |

平均成功率：

```text
reproduced DP: 4.9%
paper DP: 5.8%  # as recorded in PROGRESS.md
```

结论：9/9 任务完成 100-rollout 评测。DP 复现平均成功率 4.9%，和当前进度记录中的论文 DP 5.8% 接近。`swap_T` 低于论文，`swap_blocks` 和 `battery_try` 高于论文，其余低成功率任务符合 DP 在 RMBench 上的预期表现。

## 已知问题

历史结果缺少统一 metadata：

```text
run commit: not_recorded
train wandb id: not_recorded
eval wandb id: not_recorded
```

这批结果可以通过 checkpoint、eval_result 和 `PROGRESS.md` 定位，但不能视为完全规范化的可复现实验记录。后续复跑时应在产物 metadata 中补齐 commit、命令、batch_id 和 wandb 引用。

`swap_blocks` 和 `cover_blocks` 的 `eval_log.txt` 中只统计到 99 条 episode 输出，但 `_result.txt` 已写出最终 success rate。这里按 `_result.txt` 和 `PROGRESS.md` 采用 100-rollout 结果。
