# 通用深度学习实验规范

本文档定义深度学习实验的通用原则。目标是让实验可复现、结果可追溯，同时避免把结果目录设计得过于复杂。

## 核心原则

```text
可复现性：同一代码版本、配置、数据版本和 checkpoint 引用可以复跑。
可追溯性：训练结果和评测结果都能找到对应配置、代码版本和数据来源。
可比较性：默认实验和消融实验的差异由配置表达，而不是临时改代码。
简洁性：实验结果应集中在少数清晰目录中，不为轻量日志额外制造结果入口。
```

## 代码版本

正式实验必须从明确的 git commit 启动，启动前 `git status --short` 原则上应为空。这样 wandb、checkpoint 和 eval_result 中记录的代码版本才有实际含义。

这个要求只约束正式训练、正式评测、报告用实验和消融实验。调试性试跑可以在 dirty tree 上执行，但不应当成可复现结果引用；如果后续需要保留，应重新在 clean commit 上复跑。

当自动化系统或 AI 被要求实现代码并启动正式实验时，应先按通用代码规范完成自审和 commit，再启动实验。commit 前应确认：

```text
改动边界清楚
commit 拆分合理
没有运行产物、机器私有路径或无关改动进入 git
```

commit 完成后、正式实验启动前，还应检查最近一次 commit 的 message、diff 摘要和工作区状态：

```bash
git log -1 --pretty=full
git show --stat --oneline --format=fuller HEAD
git status --short
```

只有确认 commit 信息可读、提交内容正确、工作区干净后，才启动正式训练或评测。这样可以避免实验已经开始后才发现 commit message 或提交内容不符合溯源要求。

## 结果目录

推荐只保留两类主要结果目录：

```text
checkpoints/
  训练结果目录。保存模型 checkpoint、训练配置、训练日志和必要 metadata。

eval_result/
  评测结果目录。保存 eval 配置、success rate、episode 结果、视频或诊断文件。
```

训练和评测通常是分开的，因此最多需要两个主要结果文件夹。不要为了 manifest、pid、queue state 或 stdout log 再创建第三类长期结果目录。

如果启动器需要运行过程中的临时文件，优先放在对应的 `checkpoints/` 或 `eval_result/` 子目录中；如果只是临时排查文件，应放在 ignored 的临时目录中。

## 共享存储入口

多机 GPU 云服务器通常区分本地盘和共享云盘：

```text
本地盘：每台机器独立，适合放 workspace 和临时文件。
共享云盘：多台机器可见，适合放 dataset、checkpoint、eval result。
```

项目应提供一个共享存储入口，例如：

```text
storage -> shared storage root
```

代码和实验配置只引用：

```text
storage/checkpoints/...
storage/eval_result/...
storage/datasets/...
```

不直接引用共享云盘真实绝对路径。迁移机器时，只需要重新建立 `storage` 入口。

## wandb

wandb 是统一查询和对比入口，不是本地结果目录。

wandb 应记录：

```text
训练配置
评测配置
代码版本
数据来源
checkpoint 路径引用
eval result 路径引用
训练/eval 指标
结果摘要
```

wandb 不应默认上传：

```text
完整 checkpoint
完整 dataset
大规模视频
大规模 cache
```

大文件由共享存储管理，wandb 只负责网页上的查询、对比和摘要展示。

## 配置

影响实验语义的内容必须进入配置，例如：

```text
模型配置
数据配置
训练超参
数据增强策略
seed
eval 设置
消融参数
```

配置应随训练结果或评测结果保存一份副本，并同步到 wandb。

消融实验应通过配置改变，而不是通过临时修改代码改变。

## 实验入口

推荐使用统一实验入口：

```text
experiments/<experiment_name>/
  README.md
  run.py
  configs/
```

这只是推荐结构。如果项目已有成熟入口，可以沿用，但需要保证：

```text
入口可发现
配置可追溯
训练结果进入 checkpoints
评测结果进入 eval_result
wandb 上能查看 config 和 summary
```

实验 README 至少说明：

```text
实验目的
数据来源
默认配置和消融项
运行命令
checkpoint 位置
eval_result 位置
如何复跑
```

## 日志

日志的作用是排查问题，不应成为单独的实验结果体系。

训练日志应尽量放在对应 checkpoint 目录中。eval 日志应尽量放在对应 eval_result 目录中。

GPU、host、环境信息如果记录在日志或 wandb config 中会更方便排查，但不是实验规范的核心要求。

## 数据和 checkpoint

dataset 和 checkpoint 不进 git，不上传 wandb。

应记录它们的可复现引用，例如：

```text
storage/datasets/<dataset_name>
storage/checkpoints/<experiment_name>/<run_name>
```

如果数据经过转换，转换配置和基础校验结果应保存到数据目录或训练配置中。
