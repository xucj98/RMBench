# RMBench 项目规范

本文档只写 RMBench 对通用规范的项目级落地约定。

相关文档：

```text
docs/general_code_guidelines.md
docs/general_experiment_guidelines.md
```

## 项目结构

RMBench 是 benchmark 环境和多个 policy 子项目的组合仓库，不是单一 Python package。

当前主要结构：

```text
envs/          仿真环境和任务
description/   instruction 和任务描述
script/        项目级辅助脚本
policy/        各类 policy 子项目，部分来自上游开源库
task_config/   任务配置
docs/          文档
```

修改上游 policy 子项目时，应顺应其原有结构。例如 pi05/openpi 的核心逻辑仍放在：

```text
policy/pi05/src/openpi/
```

不要为了 RMBench 的目录偏好重构上游 package。

## Git 与提交语言

RMBench 当前默认使用中文 commit message，优先服务于本项目主要开发者的回看、审阅和实验溯源。

如果 commit 面向上游开源库、英文 issue/PR、英文协作者，或需要进入外部社区，则使用英文。

Commit message 结构仍遵循通用代码规范；只是语言默认使用中文。

## 共享存储和软链接

RMBench 当前不重构 repo-facing 软链接。现有共享数据、checkpoint 和 eval_result 继续通过 repo 内路径访问，例如：

```text
assets/...
data/<task>
policy/<policy_name>/checkpoints
eval_result
```

这些路径在本机可能是指向 `/mnt/public/xcj/rmbench` 或 `/mnt/public3/xcj/rmbench` 的软链接。`/mnt/public` 本身是 `/mnt/public3` 的软链接，因此两种写法访问的是同一块共享盘。

迁移到另一台机器时，优先按当前软链接清单重建这些 repo-facing 入口，而不是移动大文件或重新设计目录结构。代码、配置和实验脚本应引用 repo 相对路径，不应写死 `/mnt/public...` 绝对路径。

当前软链接清单可以用下面的命令导出：

```bash
find . -maxdepth 4 -type l -printf '%p -> %l\n' | sort
```

当前这些软链接是 workspace 状态，不进入 git。后续如需自动化部署，可增加一个小的初始化/检查脚本来重建软链接；脚本可以进 git，但不要把机器私有的绝对软链接本身提交成项目规范。

## 结果目录

RMBench 沿用简单的结果结构：

```text
policy/<policy_name>/
  policy 相关训练数据、processed data 和 checkpoint。

eval_result/
  评测结果、success rate、rollout 诊断和视频。
```

训练日志、训练配置、wandb id 等训练相关文件应放在对应 checkpoint 目录下。eval 日志、eval 配置和 result summary 应放在对应 eval_result 目录下。

不要额外创建长期 `runs/` 结果目录。临时 pid、queue state 或启动日志如果需要保存，应放在对应训练或评测结果目录中，或者放在 ignored 的临时目录中。

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

同一批实验如果只是 task、seed、batch size、GPU、学习率小范围覆盖或运行名不同，应优先复用同一个 `train_config_name`，通过 CLI override 或批量启动脚本传入 `repo_id`、`exp_name`、`batch_size` 等运行参数。这里的 `repo_id` 指 LeRobot 数据集名或数据集引用，不是代码仓库。不要为每个 task 都新增一个 config，导致目录退化为：

```text
checkpoints/<task_config>/<task_exp>/...
```

这种写法会让第一层和第二层重复表达同一个 run，浪费 openpi 已经提供的两层结构。

只有模型结构、policy 行为、数据 transform、state/action schema、归一化规则或默认数据语义发生变化时，才应新增 `train_config_name`。例如 LoRA 与 full finetune 可以是不同 config；同一 full finetune 只是 batch size 从 8 改到 32，不应新建 config。

pi05 的 assets 目录按 `train_config_name` 和 `repo_id` 组织：

```text
policy/pi05/assets/<train_config_name>/<repo_id>/norm_stats.json
```

`norm_stats` 反映的是某个数据集在有效数据处理链路下进入模型前后的统计量，不反映 `exp_name`、GPU 或 batch size。单纯修改 batch size、GPU、运行名或补跑编号，应复用已有 assets。

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

不要默认使用旧脚本的环境视角层级作为新实验主结构：

```text
eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
```

每个 eval run 目录应聚合同一次评测的结果、日志、配置和视频：

```text
eval_result/<batch_id>/<run_id>/
  _result.txt
  eval_log.txt
  stdout.log
  config.yaml
  episode0.mp4
  ...
```

`_result.txt` 保存最终指标；`eval_log.txt` 保存 episode 级结果；`stdout.log` 保存该次 eval 进程的 stdout/stderr；`config.yaml` 保存 deploy config、CLI overrides、task config 和脚本解析后的最终有效配置快照。单次 eval 的 stdout/stderr 是 per-run 日志，应和该 run 的结果放在同一目录，不要另放到 `_launch_logs/`。

只有跨多个 run 的队列调度日志才属于 batch 级日志，例如启动了哪些 run、分配到哪些 GPU、进程何时结束。此类日志可以放在 batch 目录下：

```text
eval_result/<batch_id>/_queue.log
```

也可以放在 ignored 的临时运行目录中。它不应替代 per-run 的 `stdout.log`。

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

这些文件应进入 git 管理，包括 README、命令脚本、job manifest、配置 override 和结果摘要。`experiments/` 不保存 checkpoint、dataset、视频、大规模日志或本地 wandb 运行目录。

policy 子项目中可以保留基础训练脚本、模型库代码和数据转换工具，但不要把“一键启动一组 RMBench 实验”的入口放在某个 policy 子目录里。

注意：repo 内的 `experiments/` 是实验入口目录，不是实验结果目录；不需要为它创建共享盘结果目录。

推荐结构：

```text
experiments/<batch_id>/
  README.md
  run.py / commands/ / configs/ / jobs/   # 可选，按实验需要
```

`README.md` 是核心。批量运行代码可以放在 `run.py` 或 `commands/*.sh` 中；配置 override 和 job manifest 可以放在 `configs/` 或 `jobs/` 中。目录名不强制，关键是入口可发现、命令可复现、结果可追溯。

RMBench 使用 batch / run 两层语义：

```text
batch:
  一批共同回答同一个问题或假设的实验，例如 DP Table 1 复现、Pi0.5 Table 1 复现、key-state 消融。

run:
  一个具体实验，通常是某个 policy、task、seed 和配置的一次训练及对应评测。
```

已有历史 checkpoint 和 eval_result 不需要为了实验入口而搬迁。README 直接说明当前 repo 相对路径或目录规则，例如：

```text
policy/DP/checkpoints/...
policy/pi05/checkpoints/...
eval_result/<task>/<policy>/...
```

未来新实验也可以继续沿用 policy 子项目自己的 checkpoint 目录和现有 eval_result 落点。关键是每个正式实验的 checkpoint 目录、eval_result 目录和 wandb config/summary 里能追溯 `batch_id`、run name、commit、训练命令、评测命令、checkpoint 引用、eval_result 引用、wandb id 和最终指标。

当前 RMBench 的训练和评测入口通常是分离的：训练脚本负责 checkpoint 和训练 wandb，`script/eval_policy.py` 负责写 `eval_result/...`。因此不要假设所有 policy 都支持训练进程内部自动 eval。更稳妥的做法是由 `experiments/<batch_id>/` 下的入口或外层 runner 串联 train job 和 eval job，并把 train/eval 的互相引用写入产物 metadata 和 wandb。

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

smoke、启动测试、单 episode 调试、video-count 测试和未完成评测不进入正式结果汇总；如需保留，只能作为排查记录或 notes，不能混入正式复现表。

## wandb

RMBench 正式训练默认上传 wandb，作为跨机器统一查询和对比入口。正式评测应优先上传 wandb 或由 runner 同步摘要；如果当前入口不支持，就必须在 eval_result metadata 或 README 结果摘要中记录 eval_result 路径和最终指标。

上传 wandb 时统一使用：

```text
project: RMBench
group: <batch_id>
job_type: train 或 eval
name: <run_name>
```

wandb 记录 config、指标、结果摘要和 repo 相对路径引用；不上传完整 checkpoint、dataset 或大规模视频。建议至少记录：

```text
batch_id
run_name
task
policy
commit
训练/eval 命令
checkpoint_ref
eval_result_ref
success_count / test_num / success_rate
```

如果 train 和 eval 能共用同一个 wandb run，优先共用一个 `wandb_id`，这样网页上直接能查到训练曲线和最终 eval 指标。当前 `script/eval_policy.py` 默认不初始化 wandb，只写本地 `eval_result`；因此现阶段至少要在 eval_result metadata 或 README 结果摘要里记录训练 `wandb_id`、eval_result 路径和 success rate，后续可以再补一个 eval wandb job 或让 runner 把评测摘要同步到 wandb。

`experiments/` 是批次说明和启动入口，wandb 是查询和对比视图。具体实验的事实记录以 checkpoint metadata、eval_result metadata 和 wandb 为准；wandb 未上传或上传不完整时，正式结果仍应能从本地产物定位和复核。

## GPU 与渲染

除非用户特别说明，否则训练和评估不得占用 GPU0。

SAPIEN 渲染设备可通过环境变量指定：

```text
SAPIEN_RENDER_DEVICE=cuda:0
```

GPU 或 host 信息如果出现在日志或 wandb config 中会更方便排查，但不是必须单独设计结果目录。
