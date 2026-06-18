# 通用代码开发规范

本文档只定义通用原则。具体目录名、环境管理方式和部署方式由项目级规范决定。

## 核心原则

代码开发优先满足三点：

```text
可移植性：代码不绑定某台机器、账号、本地盘或云盘绝对路径。
可维护性：核心逻辑和运行入口分离，新增代码顺应已有结构。
可审阅性：源码、配置、运行产物、机器私有设置边界清楚。
```

## 仓库结构

不要强制所有项目长成同一种目录结构。判断一个代码库结构是否合理，主要看：

```text
1. 可复用核心逻辑是否在稳定模块或 package 中。
2. 入口脚本是否足够薄，只负责解析参数、加载配置、调用核心逻辑。
3. 运行产物是否和源码分离。
4. 新增代码是否尊重已有模块边界。
```

对于上游开源代码库，优先顺应原有结构，不为统一目录名做大规模重构。

## 路径与配置

代码和可提交配置中不写机器相关绝对路径，例如：

```text
/mnt/...
/root/...
/home/<user>/...
```

路径默认值应是项目相对路径，或来自项目级规范定义的统一入口。机器差异通过配置、环境变量或软链接解决，不通过修改源码解决。

配置优先级建议为：

```text
命令行参数 > 环境变量 > 配置文件 > 代码默认值
```

实验语义相关的参数应写入配置文件；机器和部署相关的参数可以用环境变量或项目级软链接约定。

## 命令执行约定

文档、commit message、实验记录中的命令，默认都应从 workspace 根目录执行。

如果命令必须在其他目录运行，应在命令开头显式切换目录，例如：

```bash
cd policy/pi05 && CUDA_VISIBLE_DEVICES=1 python scripts/train.py ...
```

这样另一个开发者复制命令时，不需要猜测工作目录。

## 依赖管理

依赖应有单一可信来源，例如 `pyproject.toml`、`setup.py`、`requirements.txt`、`environment.yml` 或 Dockerfile。具体项目选择哪一种不重要，重要的是文档中说清楚：

```text
如何安装环境
依赖变更改哪里
训练/部署环境如何复现
```

不要长期维护多份互相矛盾的依赖列表。

## Git 管理

应提交：

```text
源码
配置
文档
实验入口脚本
小型 schema / metadata 模板
测试或校验脚本
```

不应提交：

```text
checkpoint
dataset
cache
日志
pid 文件
本地 wandb/mlflow 目录
虚拟环境
机器私有软链接
临时资料
```

提交前应检查：

```bash
git status --short
git diff --stat
```

并明确哪些文件应该提交、忽略或删除。不要在未确认的情况下 revert 他人或用户已有改动。

## Commit 规范

一个 commit 应只表达一个清晰意图。它可以修改多个文件，但这些文件应服务于同一个目的；如果同时包含模型改动、实验入口、bugfix、文档整理，应优先拆成多个 commit。

Commit message 建议使用：

```text
<short subject>

Motivation:
一两句话说明为什么需要这个改动。对于 fix 类提交，应说明不修改会触发什么 bug、这个 bug 是如何观察到的，或对应哪个 issue。

Changes:
- 具体改动 1
- 具体改动 2
```

Bugfix、兼容性修复或 regression fix 类提交需要额外提供 `Evidence`，写清楚能复现问题的命令、修复前现象和修复后现象。理想情况下，同一环境中 git checkout 到修复前会复现 bug，checkout 到修复后 bug 消失。

非 fix 类提交不需要写 `Evidence`。普通功能、数据链路、实验入口等提交只保留 `Motivation` 和 `Changes` 中。

小型文档、README、ignore 或格式整理提交可以只写清楚 subject，不需要 `Motivation` 和 `Changes`。

多行 commit message 必须使用 `-F`，不要使用 `-m` 拼接多行正文，也不要在 shell 普通引号中写字面量 `\n`。`-m` 只用于单行 subject。

```bash
git commit -F /tmp/commit_message.txt
```

也可以从标准输入传入 message：

```bash
cat > /tmp/commit_message.txt <<'EOF'
短标题

Motivation:
一两句话说明动机。

Changes:
- 具体改动 1
- 具体改动 2
EOF

git commit -F /tmp/commit_message.txt
```

提交完成后应立即检查最近一次提交，确认 message 格式、提交文件和 diff 摘要符合预期：

```bash
git log -1 --pretty=full
git show --stat --oneline --format=fuller HEAD
git status --short
```

如果检查发现 commit message 格式错误，但该 commit 尚未用于正式实验、训练或对外共享，可以用 `git commit --amend` 修正。若已经用于正式实验或被他人依赖，应优先保留 commit id。

示例：

```text
Fix train data loader call for current openpi API

Motivation:
Training exits at startup because scripts/train.py passes num_workers to
create_data_loader, while the current local openpi data_loader API reads
config.num_workers internally and does not accept that keyword.

Changes:
- Remove the redundant num_workers argument from scripts/train.py.

Evidence:
- Before:
  cd policy/pi05 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src .venv/bin/python scripts/train.py \
    pi0_aloha_put_back_block_key_state_default_lora \
    --exp-name=tmp_key_state_startup_check \
    --checkpoint-base-dir=storage/checkpoints/pi0 \
    --overwrite

  Observed failure:
  TypeError: create_data_loader() got an unexpected keyword argument 'num_workers'

- After:
  The same command initializes the data loader and enters train state initialization.
```
