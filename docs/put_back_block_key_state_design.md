# Put Back Block Key-State Memory 设计文档

## 1. 目的

本实验的目的不是单纯给数据集加 annotation，而是验证一个明确的假设：

```text
在 put_back_block 这种需要记住早期信息的任务上，给 pi0 增加外部 key-state memory 输入，
并让 pi0 在动作输出中同时预测下一步 memory，可以提升单帧观察策略的成功率。
```

`put_back_block` 的关键困难是：block 最初在哪个 mat 上只在早期容易从图像判断；block 被移到中心、按完按钮之后，单帧图像通常无法可靠恢复“应该放回哪个 mat”。因此模型需要一个可持续传递的 memory。

第一版要求尽量小改 pi0：不改 pi0 主体网络，不改 LeRobot 的采样器，只扩展 `observation.state` 和 `action` 的维度，并在数据转换、transform、deploy policy 中处理 key state。

## 2. V1 范围

V1 只做 `put_back_block`，使用已生成的 50 条 `demo_clean_state` 轨迹：

```text
/mnt/public3/xcj/rmbench/state_augmented_data/put_back_block/demo_clean_state
```

V1 必须实现：

```text
1. 从 RMBench hdf5 + scene_info.json + language_annotation.json 生成 LeRobot 数据集。
2. 在 LeRobot 的 observation.state 和 action 中加入 phase_id 与 mat_id。
3. 训练一个 key-state pi0。
4. eval 时维护外部 memory，并用模型输出更新 memory。
```

V1 不做：

```text
1. 不修改 pi0/openpi 的 DataLoader sampler。
2. 不复制 episode 或 frame 区间来人为控制采样比例。
3. 不做 wrong mat 输入纠错。
4. 不引入 pose、物体坐标等额外状态。
5. 不做 chunk-level repeated key-state target，即旧讨论里的 Scheme A。
```

后续消融可以生成不同的 LeRobot dataset variant，但每个 variant 仍然保持“每个原始 frame 对应一个训练 frame”的简单结构。

所有新增行为都必须由 config 控制，不要写死在代码里。V1 可以固定使用一组默认 config，但 phase / mat 的输入策略、boundary jitter、lag recovery、`W_mat` 范围等消融项都应作为 converter 或训练/eval config 参数暴露出来，方便后续批量实验和溯源。

## 3. Key State 定义

需要两个 key state。

`phase_id` 表示任务阶段：

```text
0: move_block_to_center
1: press_button
2: move_block_back_to_origin_mat
```

`mat_id` 表示 block 最初所在 mat：

```text
0: unknown
1: left
2: right
3: front
4: back
```

注意：`scene_info.json` 里当前 `origin_mat_id` 是 `0..3`，对应 `left/right/front/back`。转换到 key state 时需要加 1：

```text
key_mat_id = scene_info["origin_mat_id"] + 1
```

两个 state 都用 one-hot 表示：

```text
phase_onehot: 3 dims
mat_onehot:   5 dims
```

## 4. pi0 State/Action 布局

pi0 默认 `action_dim = 32`，现有 ALOHA 机器人只使用前 14 维。V1 使用剩余维度放 key state：

```text
dim 0:14   robot qpos / robot action
dim 14:17  phase one-hot
dim 17:22  mat one-hot
dim 22:32  zero padding
```

单帧输入：

```text
observation.state[f] =
  concat(robot_qpos[f], phase_input[f], mat_input[f], zeros(10))
```

单步 action label：

```text
action[f] =
  concat(robot_action[f], phase_after_action[f], mat_after_action[f], zeros(10))
```

其中 key-state action label 表示“执行当前 action 后，下一帧应该使用的 memory”。如果当前 query frame 是 `t`，LeRobot 取到的 action chunk 第 `k` 步就是：

```text
actions[k, 14:17] = phase_gt[t + k + 1]
actions[k, 17:22] = mat_gt[t + k + 1]
```

这样 key-state 输出是 per-step trajectory，而不是整个 chunk 共享一个值。推理时实际执行 `N = pi0_step` 步，就读取第 `N - 1` 个 key-state 预测来更新下一次 query 的 memory：

```text
next_memory = decode(actions[N - 1, 14:22])
```

这个设计不和固定 action horizon 绑定。如果之后 `pi0_step` 从 50 改成 20，memory update 仍然有明确含义。

## 5. Phase Ground Truth

从 `language_annotation.json` 的 segment frame 数得到 phase 边界。这里的 segment 指数据生成时每次调用 `self.move(...)` 记录的一个低层动作片段；`Base_Task` 会把每个片段的 language annotation 和持续帧数写入 `language_annotation.json`。因此 segment 不是人工后处理切出来的，而是原始 demo 生成过程中的低层动作边界。

`put_back_block` 一条 episode 有 10 个低层 segment：

```text
segment 0: 右臂接近并抓取 block
segment 1: 右臂抬起 block
segment 2: 右臂把 block 放到中心位置
segment 3: 左臂接近并接触 button
segment 4: 左臂向下按 button
segment 5: 左臂向上离开 button
segment 6: 左臂回到初始位置
segment 7: 右臂在中心位置重新抓取 block
segment 8: 右臂抬起 block
segment 9: 右臂把 block 放回原始 mat
```

对应到 3 个 high-level phase：

```text
segments 0..2 -> move_block_to_center
segments 3..6 -> press_button
segments 7..9 -> move_block_back_to_origin_mat
```

定义：

```text
b01 = sum(length(segment 0..2))
b12 = sum(length(segment 0..6))
L   = hdf5 episode frame count
```

逐帧 phase：

```text
phase_gt[f] = 0, if 0 <= f < b01
phase_gt[f] = 1, if b01 <= f < b12
phase_gt[f] = 2, if b12 <= f < L
```

实现要求：

```text
1. segment length 总和应和 hdf5 frame count 一致，或只差最后一帧的 action 对齐。
2. 如果不一致，converter 必须打印 episode id、hdf5 长度、segment 总长。
3. key-state label 的长度必须和现有 robot action 的长度完全一致。
4. 不允许为了 key state 单独改变 robot action 的时间对齐。
```

## 6. Mat Ground Truth 和输入规则

`mat_gt[f]` 对整条 episode 都是同一个值：

```text
mat_gt[f] = key_mat_id
```

`mat_input[f]` 的目的不同：它模拟推理时传入模型的 memory。

V1 使用一个确定性 early acquisition 窗口，而不是复制数据或修改 sampler。默认窗口只覆盖 segment 0：

```text
W_mat_end = length(segment 0)
```

原因是 `mat_id` 的估计应该发生在任务最早期：segment 0 中 block 仍在原始 mat 上，图像中能直接看到“block 来自哪个 mat”。进入 segment 1 后，block 已经被抬起，后续图像对 origin mat 的直接证据变弱；因此 V1 不默认加 margin。

如果后续要验证更宽的 acquisition 窗口，可以通过 config 开启 margin：

```text
W_mat_end = length(segment 0) + wmat_margin_frames
default wmat_margin_frames = 0
```

输入规则：

```text
if f < W_mat_end:
  mat_input[f] = unknown
else:
  mat_input[f] = key_mat_id
```

含义：

```text
1. 早期窗口内，模型必须从单帧图像预测 origin mat。
2. 早期窗口后，origin mat 被视为已经进入 memory，后续只需要保持。
3. 早期窗口后不把 mat_input 随机改成 unknown，因为后期图像通常不支持重新估计 origin mat。
```

这解决了“只有第一帧 unknown -> origin_mat，监督太少”的问题，同时不需要引入 episode copy。是否要在早期窗口内混合 known/unknown 输入，作为后续消融处理。

## 7. Phase 输入规则

V1 默认使用 ground-truth phase 作为输入 memory：

```text
phase_input[f] = phase_gt[f]
```

phase 输出仍然按 per-step target 构造：

```text
phase_after_action[f] = phase_gt[f + 1]
```

当 action chunk 跨过 phase 边界 `T` 时，target chunk 会自然变成：

```text
old, old, ..., old, new, new, ...
```

模型不需要用这个 key-state trajectory 改变当前已经生成的动作。它的作用是给下一次 query 更新 memory。

## 8. LeRobot 训练样本构造

openpi/pi0 当前对 LeRobot 的采样方式是：

```text
LeRobotDataset.__len__ = num_frames
LeRobotDataset.__getitem__(idx) 读取一个 frame idx
delta_timestamps 根据 action_horizon 从 action 字段取未来 action chunk
torch DataLoader 对 frame-level dataset shuffle
```

因此 V1 不控制“某类样本比例”。converter 只负责给每个原始 frame 生成确定的 state/action 字段。采样比例自然由原始轨迹长度决定。

对每条 episode，converter 做：

```text
for each frame f:
  observation.state[f, 0:14]  = existing robot qpos[f]
  observation.state[f, 14:17] = onehot(phase_input[f])
  observation.state[f, 17:22] = onehot(mat_input[f])
  observation.state[f, 22:32] = 0

  action[f, 0:14]  = existing robot action[f]
  action[f, 14:17] = onehot(phase_gt[f + 1])
  action[f, 17:22] = onehot(mat_gt[f + 1])
  action[f, 22:32] = 0
```

最后一帧如何处理必须沿用现有 converter 的 robot action 规则。如果现有流程会丢掉最后一帧 action，key-state action 也一起丢；如果现有流程会 repeat 最后一帧 action，key-state 也 repeat 最后一帧 gt。不要为 key state 引入新的长度规则。

## 9. 需要改的代码

需要新增或修改以下模块。

### 9.1 数据转换

建议新增一个直接 converter：

```text
RMBench hdf5 + scene_info.json + language_annotation.json -> LeRobot
```

不要先转 ALOHA 再转 LeRobot。这样可以直接写入 32 维 `observation.state` 和 `action`，也避免在中间格式里塞额外 key state。

converter 配置至少包含：

```text
source_dir
repo_id
wmat_margin_frames
phase_input_policy = gt
mat_input_policy = unknown_until_wmat_end
key_output_mode = per_step
```

V1 固定：

```text
phase_input_policy = gt
mat_input_policy = unknown_until_wmat_end
wmat_margin_frames = 0
key_output_mode = per_step
```

`key_output_mode = per_step` 是 V1 唯一实现目标。它可以直接用 LeRobot 的 per-frame `action` 字段表达。

### 9.2 pi0 transforms

现有 `AlohaInputs` / `AlohaOutputs` 默认按 14 维机器人 state/action 写：

```text
AlohaInputs 会对整个 actions 调 _encode_actions_inv
AlohaOutputs 会裁剪 actions[:, :14]
_joint_flip_mask 长度是 14
DeltaActions mask 也只应该作用在机器人关节维度
```

因此必须新增 key-state aware transform，而不是直接复用原 transform。

要求：

```text
1. state 前 14 维走 ALOHA joint/gripper 转换，14:32 原样保留。
2. action 前 14 维走 ALOHA action 转换，14:32 原样保留。
3. DeltaActions / AbsoluteActions 只作用前 14 维中的 robot joint 维度，不作用 key-state 维度。
4. eval output 不再裁掉 14:32；必须返回 robot actions 和 key-state predictions。
```

可以实现为：

```text
KeyStateAlohaInputs
KeyStateAlohaOutputs
LeRobotAlohaKeyStateDataConfig
```

### 9.3 训练配置

新增单独 train config，避免覆盖 baseline pi0：

```text
config name: pi0_aloha_put_back_block_key_state_lora
repo_id:     put_back_block_demo_clean_key_state_v1
asset_id:    put_back_block_demo_clean_key_state_v1
model:       Pi0Config(action_dim=32, action_horizon=50, ...)
LoRA:        沿用当前 pi0 RMBench LoRA 配置
```

checkpoint 仍然按现有约定保存到 `/mnt/public3/xcj/rmbench` 下的实际目录，并从 pi05 训练目录软链接过去。

### 9.4 Eval/deploy

eval policy 需要维护外部 memory。

初始化：

```text
phase_memory = 0
mat_memory   = unknown
```

每次 query：

```text
obs.state = concat(robot_qpos, onehot(phase_memory), onehot(mat_memory), zeros)
pred = policy(obs)
execute pred[:N, 0:14]
key_pred = pred[N - 1, 14:22]
```

memory update：

```text
phase_pred = argmax(key_pred[0:3])
mat_pred   = argmax(key_pred[3:8])

phase_memory = min(max(phase_memory, phase_pred), phase_memory + 1)

if mat_memory == unknown and mat_pred != unknown:
  mat_memory = mat_pred
else:
  mat_memory = mat_memory
```

第一版不允许 mat 从一个 known mat 改成另一个 known mat。这样避免后期不可观测图像导致错误覆盖 memory。

## 10. 后续消融 Dataset Variant

这些不是 V1 必做项，也不是 sampler 比例。它们是 converter 的可选配置，每次生成一个独立 repo_id，便于溯源。

实现要求：本节所有消融项都必须通过 config 控制，不允许在 converter、transform 或 deploy policy 里写死。生成 dataset、训练和 eval 时需要把这些 config 保存到对应输出目录，至少能从 repo_id / checkpoint / eval_result 追溯到当时使用的参数。

### 10.1 Scheme A 暂不作为 V1 消融

主方案是 Scheme B：

```text
action[f, key_dims] = key state after action f
```

这个方案可以被当前 LeRobot 格式自然表达：query frame `t` 取到的 action chunk 是 `action[t], action[t+1], ...`，因此 key-state target 也自然是一条 per-step trajectory。

旧讨论里的 Scheme A 是：

```text
对同一个 query frame t，action chunk 内所有 key target 都相同，
都表示执行 pi0_step 后的 next memory。
```

这个方案不能只靠一个 per-frame `action[f, key_dims]` 字段干净表达。因为 LeRobot 会把连续 frame 的 action 拼成 chunk，如果写成：

```text
action[f, key_dims] = key_gt[f + pi0_step]
```

那么 query `t` 看到的 chunk 会变成：

```text
key_gt[t + pi0_step],
key_gt[t + 1 + pi0_step],
key_gt[t + 2 + pi0_step],
...
```

这不是“chunk 内所有 key target 相同”。

因此 Scheme A 只有两种实现方式：

```text
1. 写自定义 dataset/transform，在 LeRobot 取完 action chunk 后覆盖 key dims。
2. 修改模型/训练入口，把 chunk-level memory target 作为单独监督头或单独字段。
```

这两种都超出“小改 pi0 + 不改 sampler”的 V1 范围。第一轮实验只做 Scheme B。

### 10.2 Mat 输入策略

候选配置：

```text
mat_input_policy = unknown_first_frame_only
mat_input_policy = unknown_until_wmat_end
mat_input_policy = early_hash_mix
wmat_margin_frames = 0 / 10 / 20
```

`early_hash_mix` 如果需要，可以在 `f < W_mat_end` 内用确定性 hash 选择 unknown 或 known：

```text
use_unknown = hash(episode_id, frame_id, seed) < p_unknown
```

这仍然不复制 frame，也不修改 sampler。它只是改变每个 frame 的 input memory 构造方式。

### 10.3 Phase lag recovery

用于模拟推理时 phase memory 滞后：

```text
phase_input_policy = lag_after_boundary
lag_window_frames = 10 / 20
```

规则：

```text
if f in [T, T + lag_window_frames):
  phase_input[f] = old_phase
else:
  phase_input[f] = phase_gt[f]

phase target 和 robot action 始终使用真实当前/未来标签。
```

这个 variant 会替换该窗口内的输入 memory，而不是额外复制 recovery 样本。若未来确实需要同时保留 normal 和 lag 样本，再单独讨论是否引入 episode copy 或自定义 sampler。

### 10.4 Phase boundary jitter

用于验证 phase 边界不确定性：

```text
phase_boundary_jitter_frames = 0 / 5
```

如果开启，对每个 episode 的每个 boundary 用固定 seed 采样：

```text
T_eff = T + delta, delta in [-J, +J]
```

然后用 `T_eff` 计算 `phase_gt`。jitter 不应过大，否则 phase label 会和 robot action 明显冲突。

## 11. 验证标准

数据转换完成后必须做以下检查：

```text
1. LeRobot dataset episode 数等于 50。
2. 每条 episode 的 frame 数和源 hdf5 对齐。
3. observation.state.shape[-1] == 32。
4. action.shape[-1] == 32。
5. dim 14:17 每帧 one-hot，dim 17:22 每帧 one-hot，dim 22:32 全 0。
6. mat_id 分布和 scene_info.json 一致。
7. phase 边界和 language_annotation.json 一致。
8. action chunk 跨 boundary 时，key-state target 能出现 old -> new。
```

训练/eval 完成后至少报告：

```text
1. baseline pi0 50 rollout 成功率。
2. key-state pi0 50 rollout 成功率。
3. eval 中 mat_memory 从 unknown 变成 known 的 frame/query 分布。
4. eval 中 phase_memory 发生 0->1、1->2 转移的 frame/query 分布。
5. 失败 episode 中 memory 是否预测错误，还是 robot action 执行失败。
```

这些信息用于判断 key-state memory 是否真的解决了记忆问题，而不是只改变了训练数据格式。

## 12. 当前 pi0 Eval 结果备查

以下结果是 2026-06-01 对当前三个 pi0 LoRA 模型做的 50 rollout eval。评测配置均为：

```text
policy: pi05 deploy pi0 checkpoint
task_config: demo_clean_eval
instruction_type: unseen
checkpoint_id: 30000
eval_video_log: false
```

其中“原始 obs”指 action chunk 内按原始逻辑渲染/更新 observation；`get_obs_fast` 指 action chunk 中间调用 `TASK_ENV.get_obs_fast()`，避免重复渲染图像，只更新必要的 qpos / joint action 观测。

| task | eval variant | ckpt_setting | timestamp | success rate | success count | elapsed | elapsed source | result file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swap_blocks | 原始 obs | `pi0_swap_blocks` | `2026-06-01 16:27:39` | 0.14 | 7/50 | 10:22:26 | result timestamp -> `_result.txt` mtime | `/mnt/public3/xcj/rmbench/eval_result/swap_blocks/pi05/demo_clean_eval/pi0_swap_blocks/2026-06-01 16:27:39/_result.txt` |
| swap_blocks | `get_obs_fast` | `pi0_swap_blocks_fastobs` | `2026-06-01 18:25:41` | 0.18 | 9/50 | 1:02:51 | `.start` mtime -> `_result.txt` mtime | `/mnt/public3/xcj/rmbench/eval_result/swap_blocks/pi05/demo_clean_eval/pi0_swap_blocks_fastobs/2026-06-01 18:25:41/_result.txt` |
| swap_T | 原始 obs | `pi0_swap_T` | `2026-06-01 16:27:39` | 0.16 | 8/50 | 5:27:59 | result timestamp -> `_result.txt` mtime | `/mnt/public3/xcj/rmbench/eval_result/swap_T/pi05/demo_clean_eval/pi0_swap_T/2026-06-01 16:27:39/_result.txt` |
| swap_T | `get_obs_fast` | `pi0_swap_T_fastobs` | `2026-06-01 18:16:10` | 0.16 | 8/50 | 0:36:24 | monitor `elapsed_sec=2184` | `/mnt/public3/xcj/rmbench/eval_result/swap_T/pi05/demo_clean_eval/pi0_swap_T_fastobs/2026-06-01 18:16:10/_result.txt` |
| put_back_block | 原始 obs | `pi0_put_back_block` | `2026-06-01 16:27:39` | 0.08 | 4/50 | 4:55:34 | result timestamp -> `_result.txt` mtime | `/mnt/public3/xcj/rmbench/eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block/2026-06-01 16:27:39/_result.txt` |
| put_back_block | `get_obs_fast` | `pi0_put_back_block_fastobs` | `2026-06-01 18:25:41` | 0.08 | 4/50 | 0:29:33 | `.start` mtime -> `_result.txt` mtime | `/mnt/public3/xcj/rmbench/eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_fastobs/2026-06-01 18:25:41/_result.txt` |

汇总：

```text
swap_blocks:    0.14 -> 0.18 with get_obs_fast
swap_T:         0.16 -> 0.16 with get_obs_fast
put_back_block: 0.08 -> 0.08 with get_obs_fast

swap_blocks eval time:    10:22:26 -> 1:02:51 with get_obs_fast
swap_T eval time:         5:27:59  -> 0:36:24 with get_obs_fast
put_back_block eval time: 4:55:34  -> 0:29:33 with get_obs_fast
```

`get_obs_fast` 主要降低 eval 时间，不应改变策略输入的单帧图像语义；从当前结果看，swap_T 和 put_back_block 成功率完全一致，swap_blocks 有 2/50 的差异，需要按随机性或仿真观测细节差异处理。

原始 obs 的 eval 没有单独记录 `start_time` 文件；这里的耗时由结果目录 timestamp 到 `_result.txt` 文件 mtime 推算。`get_obs_fast` 中 swap_T 使用 monitor 的 `elapsed_sec`，swap_blocks 和 put_back_block 使用 `.start` 文件 mtime 到 `_result.txt` 文件 mtime 推算。
