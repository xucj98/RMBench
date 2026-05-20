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
| DP 评估 (100 rollouts) | 🔄 进行中 | 9 任务并行，GPU 5-7；已优化 eval 速度 (5x+) |
| Mem-0 数据处理 (lerobot) | ✅ 完成 | 9/9 完整 |
| Mem-0 norm_stats | ✅ 完成 | 9/9 已生成 |
| Mem-0 环境搭建 | ✅ 完成 | PyTorch 2.6 + deepspeed + lerobot |
| Mem-0 Qwen3-VL-2B 下载 | ✅ 完成 | 迁移到 `/mnt/public3/`，双软链 |
| Mem-0 Qwen3-VL-8B 下载 | ✅ 完成 | 同上 |
| Mem-0 flash-attn 2.6.1 | ❌ 待安装 | |
| Mem-0 训练配置 | ✅ 就绪 | `execution_module_train_{task}.yaml` × 9 |
| Mem-0 Execution Module 训练 | ⏳ 待开始 | |
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

### 6. curobo 多进程 spawn 死锁（已验证：根本原因是 warp 版本）
- **最初诊断**: 评估启动后 CPU 50% 但 GPU 0%，30 分钟无进展
  - `communication_flag=True` → 启动 2 个 spawn worker 进程
  - 怀疑 worker 的 CUDA/warp 初始化在 spawn 模式下卡死
- **排查过程**: 在 `robot.py` 强制 `communication_flag = False`，让 planner 在主进程运行
- **后续验证** (2026-05-20): 修复 warp 0.15.1 后，测试发现原始多进程代码完全正常
  - 无修复版本 `plan_grippers` 返回正常（31s warmup）
  - 完整 eval 2 个 rollout 正常完成，无死锁
- **真实原因**: 当时 warp 1.13.0 缺少 `wp.torch` 子模块，导致 worker 进程 import 失败后卡住，而非 spawn 本身的问题
- **最终处理**: `communication_flag = False` 修复已回退，原始代码保持不变

### 7. deploy_policy.py 路径不匹配
- **现象**: `FileNotFoundError: ./policy/DP/checkpoints/observe_and_pickup-default-50-0/600.ckpt`
  - 代码用 `ckpt_setting`（值为 `default`）拼路径
  - 实际目录名用 `task_config`（值为 `demo_clean`）
- **解决**: 修改第 19 行 `ckpt_setting` → `task_config`

### 8. Mem-0 数据不完整
- **现象**: battery_try 6/50, put_back_block 28/50, cover_blocks 因磁盘满崩溃
- **解决**: 重新处理全部数据，9/9 完整

### 9. DP eval 5 集后崩溃 (eval_video_ffmpeg)
- **现象**: 第 6 个 rollout `AttributeError: 'NoneType' object has no attribute 'stdin'`
- **原因**: 只保存前 5 集视频，第 6 集 ffmpeg 未初始化，但 `take_action` 和 `check_success` 中未检查
- **解决**: `_base_task.py` 初始化 `self.eval_video_ffmpeg = None`，两处 guard 加 `is not None`
- **关联改动**: `eval_policy.py` 中 `save_video = ... and TASK_ENV.test_num < 5` 限制视频数量

### 10. eval_policy.py instruction 修复
- **现象**: non-expert 模式下执行了 expert-only 的 instruction 生成（调用 `generate_episode_descriptions`）
- **解决**: 加 `if expert_check: ... else: set_instruction("")` 分支

### 11. wandb 训练日志启用
- **现象**: 训练曾配置 `wandb_mode=online`，但 `wandb.init()` 在代码中被注释，训练未上传
- **解决**: 
  - `robotworkspace.py`: 取消注释 `wandb.init()`，加 `import wandb`，在 epoch 结束处加 `wandb.log()`
  - `robot_dp_14.yaml` / `robot_dp_16.yaml`: project 改名为 `RMBench`，run name 简化为 `DP_${task_name}`

### 12. Mem-0 数据处理断点续传
- **现象**: 数据处理脚本重复运行会覆盖已有数据集
- **解决**: `M1/Mn_dataset_to_lerobot.py` 加 skip 逻辑，已存在数据集不再处理

### 13. Qwen3-VL 模型管理
- **现象**: 2B 模型有两份拷贝（`policy/Mem-0/checkpoints/` 和 `checkpoints/`），占双倍空间
- **解决**: 迁移到 `/mnt/public3/xcj/rmbench/checkpoints/`，两处均建软链接指向同一物理副本
- **8B 模型**: 同样处理，通过 huggingface-cli 从 HF 下载

### 14. robot_image_dataset.py 图像宽高互换
- **现象**: 原始代码 `head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255`
  - zarr 数据为 CHW (3, 240, 320)；`moveaxis(-1, 1)` 输出 (3, 320, 240)，宽高互换
  - eval 路径 `encode_obs` 输出 (3, 240, 320)，与训练不一致
- **解决**: 移除 `np.moveaxis`，直接 `/255`

---

## 当前阻塞问题

### DP eval 速度过慢 (部分解决)
- **现象**: 100 rollout 需要 1-2 天，每个 episode ~20-40 分钟
- **根因分析** (2026-05-20):
  - SAPIEN Vulkan 光追 `take_picture()` 每个相机固定 ~165ms（不受 spp/denoiser 影响）
  - 3 相机 (head+left+right) = 500ms/帧
  - 每个 DP 模型调用周期中，中间 6 次 `get_obs()` 冗余渲染占 **65%** 时间
  - DP 推理 DDPM 100 步占 **8%**（467ms/次）
  - 实际跑 35h：swap_T 84/100, battery_try 49/100
- **优化方案**:
  1. **DDIM 10 步替代 DDPM 100 步**: 模型推理 467ms→47ms (**10x**)
     - `diffusion_unet_image_policy.py`: 新增 `set_inference_config(num_inference_steps, use_ddim)` 方法
     - `dp_model.py`: 通过 `policy.set_inference_config()` 调用，而非从外部替换 scheduler
  2. **跳过中间 get_obs 渲染**: `_base_task.py`: 新增 `get_obs_fast()` 复用缓存图像，只更新 agent_pos
     - 中间 6 次渲染 3708ms→~10ms (**370x**)
  3. `deploy_policy.yml`: 新增 `ddim_steps: 10` 配置项
     - `eval_policy.py`: `test_num` 改为可配置（支持 `--test_num 20` 做小样本验证）
- **实测加速**: 原始每卡 2-3 进程共享 GPU → 优化后每卡 1 进程独占。考虑 GPU 争抢因素，纯代码优化加速约 **5x**
  - swap_T: 25.7min/ep → 1.95min/ep (含 GPU 去争抢)
  - battery_try: 42.5min/ep → 3.5min/ep (含 GPU 去争抢)
- **性能验证** (2026-05-20, 各 20ep):
  - swap_T baseline 11.0% → 优化后 15.0% (3/20)
  - battery_try baseline 20.0% → 优化后 15.0% (3/20)
  - 差异在统计误差内，优化未影响模型性能
- **参考**: RoboTwin #83 报告了 SAPIEN 渲染在某些 seed 下异常慢（pencil objaverse 模型问题），SAPIEN #171 Open（take_picture 卡死 bug）。

### DP 评估当前进展

| 任务 | 步数 | 完成 | 成功 | 论文 |
|------|------|------|------|------|
| observe_and_pickup | 250 | 100/100 ✅ | 1 (1%) | 1% |
| put_back_block | 500 | 54/100 | 0 | 0% |
| rearrange_blocks | 700 | 39/100 | 0 | 0% |
| swap_T | 600 | 50/100 | 7 (14%) | 20% |
| battery_try | 1000 | 31/100 | 7 (23%) | 10% |
| swap_blocks | 1000 | 28/100 | 4 (14%) | 11% |
| cover_blocks | 1500 | 18/100 | 0 | 0% |
| press_button | 1500 | 18/100 | 0 | 0% |
| blocks_ranking_try | 3500 | 8/100 | 0 | 10% |

> 速率 ~0.5 step/s/任务（3 GPU 共享，每 GPU 3 任务），瓶颈在 TOPP + SAPIEN 仿真 + curobo 规划

---

## 关键文件修改记录

| 文件 | 修改内容 |
|------|---------|
| `policy/DP/diffusion_policy/model/common/lr_scheduler.py` | 修复 diffusers.optimization 导入 |
| `policy/DP/diffusion_policy/dataset/robot_image_dataset.py` | 移除多余 moveaxis |
| `policy/DP/diffusion_policy/config/task/default_task_14.yaml` | action_dim=14 |
| `policy/DP/deploy_policy.py:19` | ckpt_setting → task_config |
| `auto_eval_dp.sh` | 加 LD_LIBRARY_PATH，用 GPU 2 |
| `policy/DP/diffusion_policy/policy/diffusion_unet_image_policy.py` | 新增 `set_inference_config()` 方法，支持 DDIM 切换 |
| `policy/DP/dp_model.py` | 移除 `_enable_ddim()` hack，改用 `policy.set_inference_config()` |
| `policy/DP/deploy_policy.py` | `eval()` 中间步骤用 `get_obs_fast()` 替代 `get_obs()`；`get_model()` 支持 `ddim_steps` |
| `policy/DP/deploy_policy.yml` | 新增 `ddim_steps: 10` 配置项 |
| `envs/_base_task.py` | 新增 `get_obs_fast()` 方法：跳过渲染，复用缓存图像，只更新 agent_pos |
| `script/eval_policy.py` | `test_num` 可配置；5 视频限制；instruction 分支修复 |
| `policy/DP/diffusion_policy/workspace/robotworkspace.py` | 启用 wandb.init + wandb.log |
| `policy/DP/diffusion_policy/config/robot_dp_14.yaml` | wandb project→RMBench, run name `DP_${task_name}` |
| `policy/DP/diffusion_policy/config/robot_dp_16.yaml` | 同上 |
| `policy/Mem-0/scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py` | 跳过已存在数据集 |
| `policy/Mem-0/scripts/hdf5_to_lerobot/Mn_dataset_to_lerobot.py` | 同上 |

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
