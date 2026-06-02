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

## 共享存储入口

RMBench workspace 下应维护一个共享存储入口：

```text
storage -> /mnt/public3/xcj/rmbench
```

项目内代码、配置、实验脚本只能通过 `storage/...` 访问共享存储，不直接写 `/mnt/public3/xcj/rmbench/...`。

推荐子目录：

```text
storage/datasets/
storage/checkpoints/
storage/eval_result/
storage/cache/
```

如果历史代码需要兼容软链接，软链接应指向 `storage/...` 的相对路径，而不是直接指向 `/mnt/...`。

## 结果目录

RMBench 沿用简单的结果结构：

```text
storage/checkpoints/
  训练结果和 checkpoint。

storage/eval_result/
  评测结果、success rate、rollout 诊断和视频。
```

训练日志、训练配置、wandb id 等训练相关文件应放在对应 checkpoint 目录下。eval 日志、eval 配置和 result summary 应放在对应 eval_result 目录下。

不要额外创建长期 `runs/` 结果目录。临时 pid、queue state 或启动日志如果需要保存，应放在对应训练或评测结果目录中，或者放在 ignored 的临时目录中。

## 实验入口

RMBench 的项目级一键实验入口统一放在：

```text
experiments/<experiment_name>/
```

policy 子项目中可以保留基础训练脚本、模型库代码和数据转换工具，但不要把“一键启动一组 RMBench 实验”的入口放在某个 policy 子目录里。

推荐结构：

```text
experiments/<experiment_name>/
  README.md
  run.py
  configs/
```

## wandb

RMBench 实验默认上传 wandb，作为跨机器统一查询和对比入口。

wandb 记录 config、指标、结果摘要和 `storage/...` 路径引用；不上传完整 checkpoint、dataset 或大规模视频。

## GPU 与渲染

除非用户特别说明，否则训练和评估不得占用 GPU0。

SAPIEN 渲染设备可通过环境变量指定：

```text
SAPIEN_RENDER_DEVICE=cuda:0
```

GPU 或 host 信息如果出现在日志或 wandb config 中会更方便排查，但不是必须单独设计结果目录。
