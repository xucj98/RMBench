# RMBench 项目规范

本文档只写 RMBench 对通用规范的项目级落地约定。

相关文档：

```text
docs/general_code_guidelines.md
docs/general_experiment_guidelines.md
```

## 项目结构

RMBench 是 benchmark 环境和多个 policy 子项目的组合仓库。

当前主要结构：

```text
envs/          仿真环境和任务
description/   instruction 和任务描述
script/        项目级辅助脚本
policy/        各类 policy 子项目，来自上游开源库
task_config/   任务配置
docs/          文档
```

## Git 与提交语言

RMBench 当前默认使用中文 commit message，优先服务于本项目主要开发者的回看、审阅和实验溯源。

Commit message 结构需遵循通用代码规范。

## 共享存储和软链接

RMBench 当前不重构 repo-facing 软链接。现有共享数据、checkpoint 和 eval_result 继续通过 repo 内路径访问，例如：

```text
assets/...
data/<task>
policy/pi05/checkpoints
eval_result
```

这些路径在本机可能是指向 `/mnt/public/xcj/rmbench` 或 `/mnt/public3/xcj/rmbench` 的软链接。`/mnt/public` 本身是 `/mnt/public3` 的软链接，因此两种写法访问的是同一块共享盘。

代码、配置和实验脚本应引用 repo 相对路径，不应写死 `/mnt/public...` 绝对路径。


当前这些软链接是 workspace 状态，不进入 git。后续如需自动化部署，可增加一个小的初始化/检查脚本来重建软链接；脚本可以进 git，但不要把机器私有的绝对软链接进行git commit。

## 结果目录

RMBench 沿用简单的结果结构：

```text
policy/<policy_name>/
  policy 相关训练数据、processed data 和 checkpoint。

eval_result/
  评测结果、success rate、rollout 诊断和视频。
```

训练日志、训练配置、wandb id 等训练相关文件应放在对应 checkpoint 目录下。eval 日志、eval 配置和 result summary 应放在对应 eval_result 目录下。

不要额外创建结果或者日志目录。临时 pid、queue state 或启动日志如果需要保存，应放在对应训练或评测结果目录中。

### 长任务启动和日志

正式训练或正式评测通常是长任务。使用 `setsid bash -lc ... &`，不要使用 `nohup ... &`。当前环境中 `nohup` 曾出现返回 PID 后进程立即消失、日志为空的情况；`setsid` 会新建 session，更适合脱离当前工具会话继续运行。

标准手动训练启动模板如下，命令应从 workspace 根目录执行：

```bash
run_dir="policy/pi05/checkpoints/<train_config_name>"
mkdir -p "$run_dir"
setsid bash -lc 'cd policy/pi05 && env CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 PYTHONPATH=src .venv/bin/python scripts/train.py <train_config_name> --exp-name=<exp_name> ...' \
  > "$run_dir/<exp_name>.stdout.log" 2>&1 &
echo $! > "$run_dir/<exp_name>.pid"
```

标准手动评测启动模板如下：

```bash
run_dir="eval_result/<batch_id>/<run_id>"
mkdir -p "$run_dir"
setsid bash -lc 'env CUDA_VISIBLE_DEVICES=<gpu> ... python script/eval_policy.py ...' \
  > "$run_dir/stdout.log" 2>&1 &
echo $! > "$run_dir/pid"
```

`stdout.log` 是该 run 的 stdout/stderr；`pid` 用于查询进程是否仍在运行。Python 批量启动脚本可以使用 `subprocess.Popen(..., stdout=log, stderr=subprocess.STDOUT, start_new_session=True)`，语义上等价于手动 `setsid`。它同样应把 per-run `stdout.log` 和 `pid` 写到对应 checkpoint 或 eval_result run 目录；跨多个 run 的队列调度日志才可以放在 batch 目录下。

### pi05 checkpoint 命名

pi05/openpi 训练默认按两层语义保存 checkpoint：

```text
policy/pi05/checkpoints/<train_config_name>/<exp_name>/<step>
```

这两层目录都应被有效利用：

```text
<train_config_name>
  表示一组共享模型结构、数据处理、归一化资产和训练范式的稳定配置。

<exp_name>
  表示该配置下的一个具体 run，例如 task、variant、seed 或补跑编号。
```

同一批实验如果只是 task、seed、batch size、GPU、学习率小范围覆盖或运行名不同，应优先复用同一个 `train_config_name`，通过 CLI override 或批量启动脚本传入 `repo_id`、`exp_name`、`batch_size` 等运行参数。这里的 `repo_id` 指 LeRobot 数据集名或数据集引用。不要为每个 task 都新增一个 config，导致目录退化为：

```text
checkpoints/<task_config>/<task_exp>/...
```

只有模型结构、policy 行为、数据 transform、state/action schema、归一化规则或默认数据语义发生变化时，才应新增 `train_config_name`。例如 LoRA 与 full finetune 可以是不同 config；同一 full finetune 只是 batch size 从 8 改到 32，不应新建 config。

pi05 的 assets 目录按 `train_config_name` 和 `repo_id` 组织：

```text
policy/pi05/assets/<train_config_name>/<repo_id>/norm_stats.json
```

`norm_stats` 反映的是某个数据集在有效数据处理链路下进入模型前后的统计量，不反映 `exp_name`、GPU 或 batch size。单纯修改 batch size、GPU、运行名或补跑编号，应复用已有 assets。

RMBench 默认用最多 10,000 frames 计算 norm stats，命令必须显式传入
`--max-frames 10000`。除非用户明确要求全量统计，不要省略这个参数重复扫描完整数据集。
同一批对照实验应使用相同的 frame 上限；实际命令和是否复用 stats 需记录在实验 README。

原则上，以下变化需要重新计算 `norm_stats`：

```text
repo_id 变化，即换了 LeRobot 数据集。
会改变模型输入或监督分布的 state/action transform 变化。
归一化规则变化。
```

如果数据处理层面的变化很轻，例如只改变少量离散 one-hot key-state 的时序策略，且明确判断复用原统计量不会影响结论，可以复用同一份 assets；这种复用应在实验 README 或启动脚本中说明。

### eval_result 命名

新实验的 eval result 应按实验批次优先组织：

```text
eval_result/<batch_id>/<run_id>/
```

每个 eval run 目录应聚合同一次评测的结果、日志、配置和视频：

```text
eval_result/<batch_id>/<run_id>/
  _result.txt
  eval_log.txt
  stdout.log
  command.txt
  config.yaml
  episode0.mp4
  ...
```

`_result.txt` 保存最终指标；`eval_log.txt` 保存 episode 级结果；`stdout.log` 保存该次 eval 进程的 stdout/stderr；`command.txt` 保存启动命令、当前 commit 和关键环境变量；`config.yaml` 保存 deploy config、CLI overrides、task config 和脚本解析后的最终有效配置快照。

只有跨多个 run 的队列调度日志才属于 batch 级日志，例如启动了哪些 run、分配到哪些 GPU、进程何时结束。此类日志可以放在 batch 目录下：

```text
eval_result/<batch_id>/_queue.log
```


## 实验入口

RMBench 的项目级一键实验入口统一放在：

```text
experiments/<batch_id>/
```

在 RMBench 中，`experiments/` 同时承担两类职责：

```text
1. 实验定义和批量运行入口。
2. 批次级说明：这批实验在验证什么、怎么跑、产物在哪里查。
```

这些文件应进入 git 管理，包括 README、命令脚本、配置 override 和结果摘要。`experiments/` 不保存 checkpoint、dataset、视频、大规模日志或本地 wandb 运行目录。

policy 子项目中可以保留基础训练脚本、模型库代码和数据转换工具，但不要把“一键启动一组 RMBench 实验”的入口放在某个 policy 子目录里。


推荐结构：

```text
experiments/<batch_id>/
  README.md
  run.py / commands/ / configs/ / jobs/   # 可选，按实验需要
```

`README.md` 是核心。批量运行代码可以放在 `run.py` 或 `commands/*.sh` 中；新增的配置可以放在 `configs/` 中。

RMBench 使用 batch / run 两层语义：

```text
batch:
  一批共同回答同一个问题或假设的实验，例如 DP Table 1 复现、Pi0.5 Table 1 复现、key-state 消融。

run:
  一个具体实验，通常是某个 policy、task、seed 和配置的一次训练及对应评测。
```

README 至少写清楚：

```text
这批实验要回答什么问题
包含哪些 policy / task / seed / variant
训练命令或批量启动命令
评测命令或批量启动命令
checkpoint 目录或命名规则
eval_result 目录或命名规则
wandb project 和 group
最终结果摘要或结果表位置
```

smoke、启动测试不进入正式结果汇总。

## wandb

上传 wandb 时统一使用：

```text
project: RMBench
group: <batch_id>
job_type: train 或 eval
name: <run_name>
```

## GPU 与渲染

除非用户特别说明，否则训练和评估不得占用 GPU0。

SAPIEN 渲染设备可通过环境变量指定：

```text
SAPIEN_RENDER_DEVICE=cuda:0
```
