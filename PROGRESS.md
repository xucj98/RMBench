# RMBench 复现进度

## 目标
复现 RMBench 论文 (arXiv:2603.01229) Table 1 中 DP 和 Mem-0 的性能。

### 论文基准 — 逐任务成功率（50 demos / 100 rollouts）

**M(1) 单任务：**

| 任务 | DP | Mem-0 |
|------|-----|-------|
| Observe and Pick Up | 1% | 4% |
| Rearrange Blocks | 0% | 89% |
| Put Back Block | 0% | 90% |
| Swap Blocks | 11% | 67% |
| Swap T | 20% | 14% |
| **M(1) 平均** | **6.4%** | **52.8%** |

**M(n) 多任务：**

| 任务 | DP | Mem-0 |
|------|-----|-------|
| Battery Try | 10% | 28% |
| Blocks Ranking Try | 10% | 18% |
| Cover Blocks | 0% | 68% |
| Press Button | 0% | 0% |
| **M(n) 平均** | **5.0%** | **28.5%** |

> **关键**：DP 在 9 个任务中 **5 个为 0%**（Rearrange Blocks, Put Back Block, Cover Blocks, Press Button, Observe 仅 1%）。因此看到 0% 成功率不等于 bug。

---

## 整体进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| DP 训练 (9任务 × 600epoch) | ✅ 完成 | checkpoints 在 `policy/DP/checkpoints/` |
| DP 评估环境搭建 | ✅ 完成 | pytorch3d + curobo + warp 全部可用 |
| DP 评估 (100 rollouts) | 🔄 进行中 | 9 任务并行，GPU 2-7 利用率 50-100%；~50min/任务 |
| Mem-0 数据处理 (lerobot) | ✅ 完成 | 9/9 完整 |
| Mem-0 norm_stats | ✅ 完成 | 9/9 已生成 |
| Mem-0 环境搭建 | ✅ 完成 | PyTorch 2.6 + deepspeed + lerobot |
| Mem-0 模型下载 (Qwen3-VL-2B) | 🔄 进行中 | wget 下载 model.safetensors |
| Mem-0 Execution Module 训练 | ⏳ 待开始 | 需 DP 完成后释放 GPU |
| Mem-0 Planning Module 训练 | ⏳ 待开始 | |

---

## 遇到并已解决的问题

### 1. 磁盘空间不足
- **现象**: 根分区 100G，DP 训练 checkpoint 和 Mem-0 数据占满磁盘导致崩溃
- **解决**: 数据迁移到 `/mnt/public3/xcj/rmbench/`，本地创建软链接

### 2. pytorch3d 安装失败
- **现象**: `pip install pytorch3d` 源码编译失败（PyTorch 2.7 无预编译 wheel）
- **解决**: 添加 `--no-build-isolation` 参数，让构建环境能访问已安装的 torch
  ```bash
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
  ```

### 3. curobo 未安装（DP 评估阻塞）
- **现象**: 评估需要 CuroboPlanner 做运动规划，pip 源中 `nvidia-curobo` 是 1.0kB 占位包
- **解决**: 使用仓库内置 `envs/curobo/` 源码 + `--no-build-isolation` 安装
  ```bash
  cd envs/curobo && pip install -e . --no-build-isolation
  ```

### 4. warp-lang 版本不兼容
- **现象**: `AttributeError: module 'warp' has no attribute 'torch'`
  - curobo v0.7.8 需要 `wp.torch.device_from_torch()`
  - pip 安装的最新 warp 1.13.0 已移除 `wp.torch` 子模块
- **解决**: 降级到 warp 0.15.1
  ```bash
  pip install 'warp-lang>=0.9.0,<1.0.0'
  ```

### 5. curobo CUDA 扩展找不到 libc10.so
- **现象**: `ImportError: libc10.so: cannot open shared object file`
  - curobo 编译的 .so 依赖 torch 的 .so，但 torch/lib 不在 LD_LIBRARY_PATH 中
- **解决**: 运行前设置环境变量
  ```bash
  export LD_LIBRARY_PATH="/root/miniconda3/envs/RMBench/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
  ```

### 6. curobo 多进程 spawn 死锁
- **现象**: 评估启动后 CPU 50% 但 GPU 0%，30 分钟无进展
  - aloha-agilex 的 `dual_arm_embodied=True` 导致左右 curobo yml 不同
  - `communication_flag=True` → 启动 2 个 spawn worker 进程
  - worker 的 CUDA/warp 初始化在 spawn 模式下卡死
- **解决**: 在 `robot.py:307` 强制 `communication_flag = False`，让 planner 在主进程运行
  - 实测 CuroboPlanner 主进程初始化仅需 24 秒

### 7. deploy_policy.py 路径不匹配
- **现象**: `FileNotFoundError: ./policy/DP/checkpoints/observe_and_pickup-default-50-0/600.ckpt`
  - 代码用 `ckpt_setting`（值为 `default`）拼路径
  - 实际目录名用 `task_config`（值为 `demo_clean`）
- **解决**: 修改第 19 行 `ckpt_setting` → `task_config`

### 8. Mem-0 数据不完整
- **现象**: battery_try 6/50, put_back_block 28/50, cover_blocks 因磁盘满崩溃
- **解决**: 重新处理全部数据，9/9 完整

---

## 当前阻塞问题

### DP 评估成功率 0%
- **现象**: 所有 rollout 全部 `Fail! | max reward: 0`
  - DP 模型被加载、推理、执行运动规划链路全部正常
  - 但策略没有做出任何有意义的行动（max reward 恒为 0）
- **可能原因**:
  1. DP checkpoint 训练不充分（但 loss 收敛到 ~0.0001）
  2. 动作维度/缩放与期望不匹配
  3. take_action 对 action 格式/范围有特定要求
  4. max_reward 代码逻辑问题
- **待排查**: 
  - 检查 DP 模型的 get_action 输出分布
  - 确认 action 格式和缩放与 task env 一致

---

## 关键文件修改记录

| 文件 | 修改内容 |
|------|---------|
| `policy/DP/diffusion_policy/model/common/lr_scheduler.py` | 修复 diffusers.optimization 导入 |
| `policy/DP/diffusion_policy/dataset/robot_image_dataset.py` | 移除多余 moveaxis |
| `policy/DP/diffusion_policy/config/task/default_task_14.yaml` | action_dim=14 |
| `policy/DP/deploy_policy.py:19` | ckpt_setting → task_config |
| `envs/robot/robot.py:308` | 强制 communication_flag=False |
| `auto_eval_dp.sh` | 加 LD_LIBRARY_PATH，用 GPU 2 |

---

## 关键路径

| 资源 | 路径 |
|------|------|
| DP checkpoints | `/mnt/public3/xcj/rmbench/dp_checkpoints/` (本地软链 `policy/DP/checkpoints/`) |
| DP zarr data | `/mnt/public3/xcj/rmbench/dp_data/` |
| Mem-0 lerobot | `/mnt/public3/xcj/rmbench/mem0_lerobot_datasets/` |
| Mem-0 assets | `/mnt/public3/xcj/rmbench/mem0_assets/` |
| 任务演示视频 | `/mnt/public3/xcj/rmbench/task_videos/` |
| curobo 源码 | `envs/curobo/` |
| eval 输出 | `eval_result/` (软链到大硬盘) |
