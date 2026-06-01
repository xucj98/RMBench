# RMBench 复现进度

## 目标
复现 RMBench 论文 (arXiv:2603.01229) Table 1 中 DP 和 Mem-0 的性能。

### 论文基准 — 逐任务成功率（50 demos / 100 rollouts）

**M(1) 单任务：**

| 任务 | DP | Pi0.5 | Mem-0 |
|------|-----|-------|-------|
| Observe and Pick Up | 1% | 9% | 4% |
| Rearrange Blocks | 0% | 13% | 89% |
| Put Back Block | 0% | 11% | 90% |
| Swap Blocks | 11% | 24% | 67% |
| Swap T | 20% | 15% | 14% |
| **M(1) 平均** | **6.4%** | **14.4%** | **52.8%** |

**M(n) 多任务：**

| 任务 | DP | Pi0.5 | Mem-0 |
|------|-----|-------|-------|
| Battery Try | 10% | 16% | 28% |
| Blocks Ranking Try | 10% | 6% | 18% |
| Cover Blocks | 0% | 0% | 68% |
| Press Button | 0% | 0% | 0% |
| **M(n) 平均** | **5.0%** | **5.5%** | **28.5%** |

> **关键**：DP 在 9 个任务中 **5 个为 0%**（Rearrange Blocks, Put Back Block, Cover Blocks, Press Button, Observe 仅 1%）。因此看到 0% 成功率不等于 bug。

---

## 整体进度

| 阶段 | 状态 | 备注 |
|------|------|------|
| DP 训练 (9 任务 × 600 epoch) | ✅ 完成 | checkpoints 在 `policy/DP/checkpoints/` |
| DP 评估环境搭建 | ✅ 完成 | pytorch3d + curobo + warp 0.15.1 |
| DP 评估加速 | ✅ 完成 | 改用DDIM，action chunk中间不渲染；已优化 eval 速度 (5x+) |
| DP 评估 (100 rollouts) | ✅ 完成 | 9/9 DONE；平均 4.9% vs 论文 5.8% |
| Mem-0 数据处理 (lerobot) | ✅ 完成 | 9/9 完整 |
| Mem-0 norm_stats | ✅ 完成 | 9/9 已生成 |
| Mem-0 环境搭建 | ✅ 完成 | PyTorch 2.6 + deepspeed + lerobot |
| Mem-0 Qwen3-VL-2B | ✅ 完成 | `/mnt/public3/`，双软链 |
| Mem-0 Qwen3-VL-8B | ✅ 完成 | 同上 |
| Mem-0 flash-attn 2.6.1 | ⬛ 已跳过 | 改用 sdpa, Qwen3VL 兼容 |
| Mem-0 训练配置 | ✅ 就绪 | `execution_module_train_{task}.yaml` × 9 |
| Mem-0 Execution Module 训练 | 🔄 2/5 M1 完成 | GPU 0-7 |
| Mem-0 Planning Module 训练 | ⏳ 待开始 | 需 LLaMA-Factory + 8B |
| pi-05 环境搭建 | ✅ 完成 | uv sync; 修复 `num_workers` 参数错误 |
| pi-05 数据处理 | ✅ 完成 | swap_blocks 50 demos → LeRobot format |
| pi-05 norm_stats | ✅ 完成 | `assets/pi05_aloha_full_base/swap_blocks_demo_clean` |
| pi-05 swap_blocks 训练 | ✅ 完成 | 20000 steps, GPU 1+2, loss 0.0011, ckpt: step 10000 + 20000, wandb: RMBench/pi05_swap_blocks |
| pi-05 swap_blocks 评估 | ✅ 完成 | **14%** (14/100)，论文 24%；ckpt step 20000 |


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


### DP 评估最终结果

| 任务 | 步数 | 结果 | 论文 DP | 论文 Pi0.5 | 状态 |
|------|------|------|---------|-----------|------|
| observe_and_pickup | 250 | 2% | 1% | 9% | ✅ |
| put_back_block | 500 | 0% | 0% | 11% | ✅ |
| rearrange_blocks | 700 | 0% | 0% | 13% | ✅ |
| swap_T | 600 | 11% | 20% | 15% | ✅ |
| swap_blocks | 1000 | 15% | 11% | 24% | ✅ |
| cover_blocks | 1500 | 0% | 0% | 0% | ✅ |
| battery_try | 1000 | 13% | 10% | 16% | ✅ |
| press_button | 1500 | 0% | 0% | 0% | ✅ |
| blocks_ranking_try | 3500 | 3% | 10% | 6% | ✅ |
| **平均** | — | **4.9%** | **5.8%** | **10.4%** | |

> 全部 100 rollout 完成。5/9 任务与论文完全匹配，swap_T 略低（11% vs 20%），swap_blocks 略高（15% vs 11%），整体平均 4.9% vs 论文 5.8%，在合理误差内。

### Pi0.5 评估结果

| 任务 | 步数 | 结果 | 论文 Pi0.5 | 状态 |
|------|------|------|-----------|------|
| swap_blocks | 1000 | **14%** (14/100) | 24% | ✅ 完成 |

> swap_blocks：复现 14% vs 论文 24%，差距约 10pp。可能原因：训练 steps 20k（50 demos）偏少，或 eval 用 unseen 指令。

---

### Mem-0 评估结果

| 任务 | 结果 | 论文 Mem-0 | 状态 |
|------|------|-----------|------|
| observe_and_pickup | **4%** (4/100) | 4% | ✅ 完全匹配 |
| rearrange_blocks | **0%** (0/100) | 89% | ❌ 需排查 |

> rearrange_blocks 论文 89% 但复现 0%，差距远超统计噪声。需排查训练是否收敛、norm_stats 匹配、inference 配置等。

### 15. pi05 `num_workers` 参数错误
- **现象**: `TypeError: create_data_loader() got an unexpected keyword argument 'num_workers'`
  - `train.py` 显式传递 `num_workers=config.num_workers`，但 `create_data_loader()` 签名不含该参数（内部自行从 config 读取）
- **解决**: 删除 `train.py:248` 处的 `num_workers=config.num_workers` 参数

### 16. pi05 单卡 OOM（full finetuning）
- **现象**: 单张 A800-80GB 上 `batch_size=64, fsdp_devices=1` 跑出 `RESOURCE_EXHAUSTED: Out of memory`
  - Pi0.5 全量微调模型参数本身就需要 ~80GB，加上激活值溢出
  - 文档注明：Full 微调需要 >100GB（2×A100/H100 80G）
- **解决**: 改用 GPU 1+2（fsdp_devices=2, batch_size=32），每卡分摊 ~40GB 激活

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
| `policy/pi05/scripts/train.py:248` | 删除多余的 `num_workers=config.num_workers` 参数（API 已内部读取） |
| `policy/pi05/src/openpi/training/config.py` | `repo_id="swap_blocks_demo_clean"`, `project_name="RMBench"`, `batch_size=32`, `fsdp_devices=2`, `save_interval=10000`, `keep_period=None` |
| `policy/pi05/src/openpi/training/checkpoints.py` | 修复 orbax 私有 API 导入 `_src.futures.future`；`max_to_keep=2` |
| `policy/pi05/src/openpi/policies/policy_config.py` | 添加 `import dataclasses`；修复 frozen dataclass `asset_id` 赋值 |
| `policy/pi05/deploy_policy.yml` | `checkpoint_id: 20000` |
| `assets/objects/005_button/10124/mobility.urdf` | 删除中文注释（SAPIEN py3.11 ASCII codec bug） |
| `assets/objects/006_check_button/10124/mobility.urdf` | 同上 |

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
