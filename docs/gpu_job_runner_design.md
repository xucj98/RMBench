# GPU Job Runner 设计草案

本文档整理 RMBench 后续可复用 GPU queue runner 的设计。目标是让 pi0、DP、Mem-0 等不同 policy 的实验都能复用同一套 GPU 资源调度和队列管理逻辑。

## 目标

当前 put_back_block key-state 批量实验脚本把实验定义、数据准备、GPU 分配和队列管理写在一起，移植性不够好。后续应拆成：

```text
实验入口：定义这批实验是什么。
policy adapter：把实验转换成统一 job manifest。
通用 runner：只负责 GPU 资源判断、排队、启动、状态记录。
policy 训练脚本：只负责实际训练或评测。
```

通用 runner 不应该知道 pi0、DP 或 Mem-0 的内部逻辑，只认识统一格式的 job。

## 当前问题

当前脚本的主要问题：

```text
1. GPU 写进 VARIANTS，实验语义和机器资源绑定。
2. 队列逻辑只服务于 put_back_block key-state，不能直接复用到 DP 或其他 policy。
3. GPU 空闲判断如果依赖 PID，在云 GPU 容器里不可靠。
4. 只支持“GPU 空闲才启动”的二值判断，不能表达显存足够时的共置运行。
```

## 统一 Job

runner 的输入是 job manifest。每个 job 至少包含：

```json
{
  "name": "pi0_key_state_default",
  "cwd": "policy/pi05",
  "cmd": [
    ".venv/bin/python",
    "scripts/train.py",
    "pi0_aloha_put_back_block_key_state_default_lora",
    "--exp-name=pi0_put_back_block_key_state_default",
    "--checkpoint-base-dir=storage/pi0_checkpoints"
  ],
  "env": {
    "PYTHONPATH": "src"
  },
  "log_path": "storage/pi0_checkpoints/pi0_key_state/default/train.log",
  "requires_gpu": true,
  "gpu_requirements": {
    "num_gpus": 1,
    "placement": "exclusive",
    "min_free_memory_mib": 72000,
    "max_existing_util": 10,
    "stable_polls": 2
  }
}
```

DP job 使用同一结构，只是 `cwd` 和 `cmd` 不同：

```json
{
  "name": "dp_swap_blocks_seed0",
  "cwd": "policy/DP",
  "cmd": ["python", "train.py", "--config", "configs/swap_blocks.yaml"],
  "env": {},
  "log_path": "storage/checkpoints/dp/swap_blocks_seed0/train.log",
  "requires_gpu": true,
  "gpu_requirements": {
    "num_gpus": 1,
    "placement": "memory-fit",
    "min_free_memory_mib": 16000,
    "max_existing_util": 60,
    "stable_polls": 2
  }
}
```

所有路径应从 workspace 根目录解释。如果 job 需要在 policy 子目录运行，用 `cwd` 表达，不要求命令里手写 `cd ... && ...`。

## GPU 分配

实验变体里不写 GPU。GPU 由 runner 启动参数指定：

```bash
python script/run_job_queue.py --jobs experiments/xxx/jobs.json --gpus 4,5,6,7
```

也可以自动扫描：

```bash
python script/run_job_queue.py --jobs experiments/xxx/jobs.json --auto-gpus --exclude-gpus 0
```

RMBench 默认应保守排除 GPU0，除非用户显式指定使用 GPU0。

runner 给子进程设置：

```text
CUDA_VISIBLE_DEVICES=<assigned_gpu>
```

如果 job 需要 SAPIEN 渲染设备，可以通过 job env 模板显式声明：

```json
{
  "env": {
    "SAPIEN_RENDER_DEVICE": "cuda:{assigned_gpu}"
  }
}
```

## GPU 可用性判断

云 GPU 容器里经常只能看到显存和利用率，看不到其他用户或其他容器的 PID。因此：

```text
外部占用判断：基于 nvidia-smi 的 memory.used、memory.total、utilization.gpu。
本 runner 启动的 job：额外用 pid 和 return code 管理生命周期。
```

不要依赖 `nvidia-smi` 进程列表判断 GPU 是否被外部任务占用。

## Placement 策略

GPU 调度不应只有空闲/非空闲两种状态。runner 应支持三类策略：

```text
exclusive:
  只放到接近空闲的 GPU。适合 pi0 这类显存和算力占用很高的训练。

memory-fit:
  允许 GPU 上已有任务，只要剩余显存足够，并且当前 util 不超过阈值。

opportunistic:
  更激进，主要看剩余显存是否足够；适合临时试跑或低优先级任务。
```

默认使用 `exclusive`。共享 GPU 必须由 job 或 runner 参数显式启用。

典型判断逻辑：

```text
free_mem = memory.total - memory.used

eligible if:
  free_mem >= min_free_memory_mib
  utilization.gpu <= max_existing_util
  连续 stable_polls 次满足条件
```

共置运行会影响吞吐和可比性，因此 runner 应记录启动时 GPU 快照：

```text
assigned_gpu
placement
gpu_total_mem_mib
gpu_used_mem_mib
gpu_free_mem_mib
gpu_util_percent
allow_shared_gpu / placement policy
```

这些信息应进入 queue state，也可以同步到 wandb config 或 summary。

## 队列状态

runner 应维护一个状态文件，例如：

```text
queue_state.json
```

状态至少包含：

```text
pending jobs
running jobs: name, pid, assigned_gpu, start_time, log_path
succeeded jobs: name, return_code, start_time, end_time
failed jobs: name, return_code, log_path
GPU launch snapshots
runner args
```

runner 重启时应能从 state 恢复：

```text
1. 已成功的 job 不重复跑。
2. pid 仍存活的 running job 继续跟踪。
3. pid 不存在且无成功记录的 job 标记为 failed 或重新排队，由参数控制。
```

## 实验入口职责

项目级实验入口仍放在：

```text
experiments/<experiment_name>/
```

实验入口负责：

```text
1. 定义实验目的和 variants。
2. 执行数据转换、norm stats 等准备步骤，或生成对应 CPU jobs。
3. 生成统一 job manifest。
4. 调用通用 runner，或提示用户用 runner 启动。
```

实验入口不负责：

```text
1. 写死 GPU 编号。
2. 判断 GPU 是否被外部用户占用。
3. 管理跨 policy 的通用队列状态。
```

## 推荐命令

只用 GPU4-7 跑一批实验：

```bash
python script/run_job_queue.py --jobs experiments/put_back_block_key_state/jobs.json --gpus 4,5,6,7
```

自动选择可用 GPU，但排除 GPU0：

```bash
python script/run_job_queue.py --jobs experiments/put_back_block_key_state/jobs.json --auto-gpus --exclude-gpus 0
```

允许显存足够时共置：

```bash
python script/run_job_queue.py \
  --jobs experiments/dp_baseline/jobs.json \
  --gpus 4,5,6,7 \
  --placement memory-fit
```

预览将启动哪些 job，不真正运行：

```bash
python script/run_job_queue.py --jobs experiments/xxx/jobs.json --gpus 4,5,6,7 --dry-run
```

## 实现顺序

建议后续按这个顺序实现：

```text
1. 定义 job manifest schema。
2. 实现 nvidia-smi GPU snapshot parser。
3. 实现 exclusive placement 和基本队列。
4. 支持 memory-fit placement。
5. 支持 resume queue_state。
6. 将 put_back_block key-state 启动器改为生成 jobs，再调用 runner。
7. 接入 DP 实验，验证跨 policy 复用。
```

## 注意事项

GPU 空闲判断和 job 启动之间存在 race condition：其他容器可能在同一时间启动任务。runner 只能降低风险，不能提供严格互斥。严格互斥需要集群调度器或共享锁机制。

共置运行会改变训练耗时和实验可比性。正式实验如果使用 `memory-fit` 或 `opportunistic`，应在结果记录中明确保留 placement 策略和启动时 GPU 快照。
