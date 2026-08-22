# X1Pro Drawer Sorting 状态转移与逐时间戳样本设计

## 1. 目标与固定训练时序

本设计用于 `/data/x1pro_table_clean` 中 `anno/tags.json` 包含
`drawer_sorting` 的 X1Pro 主从遥操作数据。原始机器人轨迹和视频为 30 Hz，转换后的
LeRobot 数据固定为 15 Hz；Pi0.5 固定使用：

- `action_horizon = 30`，即预测未来 2 秒；
- `state_history_size = 3`；
- `state_future_size = 3`；
- `state_step = 1`；
- 默认部署 `latency_step = 3`；
- Serial memory 的训练 query stride 为 15 个 15 Hz frame，即 1 秒。

动作边界使用 soft 设计：30-step action chunk 可以跨 memory 转移边界，不在 1–6 的
标注边界截断或重复末动作。

## 2. 标注语义

`anno/sort.json` 的所有区间统一解释为半开区间 `[start, end)`，索引位于 30 Hz
原始时间轴：

| 标注 | 事件 | 对应 memory |
| --- | --- | --- |
| 1 | 观察到第一类物品 | `item_1` |
| 2 | 观察到第二类物品 | `item_2` |
| 3 | 观察到第三类物品 | `item_3` |
| 4 | 抓放第一类物品 | `item_1` |
| 5 | 抓放第二类物品 | `item_2` |
| 6 | 抓放第三类物品 | `item_3` |

这里不把 1–6 直接当作六个互斥 phase。共享 memory 只有一个字段：

```text
drawer_target ∈ {observe, item_1, item_2, item_3}
```

其中 `observe` 同时表示尚未获取下一类物品，策略应执行观察；`item_i` 表示已经知道
下一步要处理第 i 类物品，策略应在该 memory 条件下执行抓放。

合法状态转移为：

```text
observe -> {observe, item_1, item_2, item_3}
item_1 -> {item_1, observe}
item_2 -> {item_2, observe}
item_3 -> {item_3, observe}
```

不允许 `item_i -> item_j (i != j)` 直接跳转；正常任务必须先回到 `observe`。

## 3. 30 Hz 语义时间轴

记原始 frame 为 `r`，episode 长度为 `N`。先在完整 30 Hz 时间轴上构造
`m_sem[r]`：

1. episode 开始时为 `observe`；
2. 进入 1/2/3 区间的起点时，分别更新为 `item_1/item_2/item_3`；
3. 观察区间结束不清空 memory，物品类别一直锁存到对应抓放完成；
4. 进入 4/5/6 区间的起点时，再次强制更新为对应 item；
5. 到达任意 4/5/6 区间的 `end` 时，更新为 `observe`。

第 4 条是缺观察数据能够参与训练的核心规则。它也处理个别标注错配和重试：即使前面
没有匹配的 1/2/3，或者此前观察标签与当前执行标签不一致，4/5/6 区间内的行为仍然明确
解释为在对应已知 item memory 下执行的动作。

同一标签出现多个执行区间时，每个区间独立处理：每个区间起点强制 item，区间终点返回
`observe`。这保留重试或分段执行的真实行为，不把两个相隔很远的区间错误合并。

## 4. 15 Hz timestamp 映射

转换后 timestamp 记为 `k`。其对应的 30 Hz 原始索引为：

```text
r_k = clip(round(k * 30 / 15), 0, N - 1)
```

所有视频、机器人状态和 memory 标签使用同一个 `r_k`，避免视频与状态分别降采样造成
边界偏移。目标 memory 为：

```text
m_plus[k] = m_sem[r_k]
```

Serial 和 Full 共用同一个 previous-memory 定义。默认先取上一次 1 秒 query 的输出：

```text
m_minus[k] = m_plus[max(k - 15, 0)]
```

然后应用 execution input override：若 `r_k` 位于标签 4、5、6 的任一执行区间，分别令：

```text
label 4: m_minus[k] = m_plus[k] = item_1
label 5: m_minus[k] = m_plus[k] = item_2
label 6: m_minus[k] = m_plus[k] = item_3
```

因此，缺少观察片段不会产生错误的 `observe -> item_i` 分类样本；对应 frame 只训练
`item_i -> item_i` 的状态保持，以及 `item_i` 条件下的抓放动作。

此外构造逐帧 `memory_action_valid[k]`。如果某个 4/5/6 起点之前的有效 memory 并非对应
item，说明该 item 是仅凭 execution override 强制注入的；该执行区间内
`memory_action_valid=false`。这解决 Full State action chunk 从区间外跨入区间时的隐含
错误监督：这段 3D memory action 不参与 loss。直到执行结束返回 `observe` 时重新设为 true，
因此 `item_i -> observe` 完成转移仍正常学习。robot action 是否保留再由下面的 query-time
因果规则决定。

robot action 还要满足 query-time 因果性。若 chunk 起点尚未进入强制执行段，则从 chunk 中
首次遇到 `memory_action_valid=false` 起，余下 robot action loss 一并屏蔽，避免模型在
`observe` 条件下学习一个尚未获知 item 的抓放动作。若 chunk 起点已经位于强制执行段，说明
当前输入已经 override 为对应 item：当前连续的强制执行段 robot action 全部保留；执行结束后的
soft 跨界动作也保留，但若 horizon 内又遇到下一个强制执行段，则从新边界起屏蔽。

## 5. 每个 timestamp 的 SM2SM robot 监督

X1Pro robot 向量为 28D：

```text
slave/follow left  7D
slave/follow right 7D
master left        7D
master right       7D
```

在 LeRobot row `k` 中：

```text
robot_state[k]  = sm2sm[r_k]
robot_action[k] = sm2sm[r_(k+1)]
```

即状态与动作都是 SM2SM，动作是下一 15 Hz frame 的绝对主从目标。最后一个视频 frame
不产生 row。

## 6. Full State sample

Full State 使用 3D implicit-unknown one-hot，给第 32 维 inpainting availability mask 留出
位置：

```text
encode(observe) = [0, 0, 0]
encode(item_1)  = [1, 0, 0]
encode(item_2)  = [0, 1, 0]
encode(item_3)  = [0, 0, 1]
```

LeRobot 中每个 row 为：

```text
state[k, 0:28]   = robot_state[k]
state[k, 28:31]  = encode(m_minus[k])
state[k, 31]     = 0  # 原始数据中没有缺失

actions[k, 0:28]  = robot_action[k]
actions[k, 28:31] = encode(m_plus[k])
actions[k, 31]    = 0
memory_action_valid[k] = 该 memory action 是否来自真实状态转移
```

memory 输出使用当前 observation `k` 的 `m_plus[k]`，不是 `k+1`，因为它表达“看完当前
画面后应更新成什么状态”；robot action 仍是下一 frame 的动作目标。

loader 会与 30-step `actions` 同步读取 30-step `memory_action_valid`。Full State 在
`memory_action_valid=false` 的 timestep 屏蔽 28:31 三个 memory 维度；Full 与 Serial 都按
上述 query-time 因果规则构造 28D robot action mask。第 31 个 padding/inpainting mask 维度
从不参与 action loss。

## 7. Serial Soft sample

Serial Soft 复用同一个 LeRobot row，但连续 robot 向量只取前 28D，memory 使用离散 sidecar：

```text
key_state_input_ids[k]  = id(m_minus[k])
key_state_target_ids[k] = id(m_plus[k])
key_state_target_mask[k] = true
```

类别 id 固定为：

```text
observe=0, item_1=1, item_2=2, item_3=3
```

训练采用 teacher forcing：先由 observation 与 `m_minus` 预测 `m_plus`，再把 GT
`m_plus` 的 learnable embedding 放入 action block 之前，作为 30-step action 的显式条件。
推理时使用 schema 的合法转移 mask 后 argmax 得到 `m_plus`，并把它作为下一次 query 的
previous memory。

## 8. 边界的精确定义

假设某个标签 4 的区间为 `[S, T)`：

- `r < S`：按此前 memory 构造；
- `S <= r < T`：输入和输出都强制为 `item_1`；
- 若起点前没有通过标签 1 获得 `item_1`，Serial 仍把这段作为 `item_1 -> item_1`；Full
  State 则屏蔽整个区间的 memory action loss，避免跨边界 chunk 学到伪
  `observe -> item_1`；
- `r >= T`：目标为 `observe`；若 1 秒前仍处于执行段，则该 sample 是
  `item_1 -> observe` 的完成判断监督。

假设标签 1 的区间为 `[S, T)`：

- `r < S`：通常为 `observe`；
- `S <= r < T`：当前画面可观察到第一类物品，目标为 `item_1`；
- `r >= T`：`item_1` 继续锁存，直到标签 4 的结束边界。

这个定义坚持“当前看到的画面决定当前输出状态”，同时允许 action chunk 跨边界。模型不需要
在边界强制停住重新 query。特别地，`observe -> item_i` 的有效监督只来自 1/2/3 区间，
绝不来自 4/5/6 的强制输入补全。

## 9. History/Future 与 action inpainting

训练 loader 对每个 row 读取：

```text
state[k-3], state[k-2], state[k-1], state[k],
state[k+1], state[k+2], state[k+3]
```

输入 adapter 执行以下因果约束：

- history slave/master 保留，用于主从延迟建模；
- future slave 一律复制 current slave，禁止使用未来真机反馈；
- future master 保留，作为已知动作前缀进行 inpainting；
- Full State 的 future memory 一律复制 current `m_minus`，禁止未来 memory 泄漏；
- 第 31 维为 availability mask。训练中真实可用位置为 0，被 drop 或部署时未知的 future
  master 位置为 1；
- 训练增强只在训练 data config 中启用，部署 config 不随机 drop。

Pi0.5 仍把 current state 放入原有离散 prefix，同时把完整 7-frame state sequence 投影到
action suffix 前。Serial memory token 与该 state sequence 可以同时启用。

## 10. 数据可用性与 fail-fast 规则

截至 2026-08-22 的审计结果：

- `drawer_sorting` tag：134 条；
- 有 `anno/sort.json`：123 条；
- 缺至少一个观察标签：89 条；
- 三个观察标签全部缺失：18 条；
- 可通过严格格式检查且同时含 4/5/6：119 条；
- 11 条缺 `sort.json`、1 条空标注、1 条奇数长度区间标注、另 2 条缺少至少一类执行事件，
  共 15 条不进入正式数据集。

converter 对缺观察宽容，但对以下情况 fail fast 或跳过并写入审计 metadata：缺相机/轨迹、
非 30 Hz、区间越界、区间重叠、4/5/6 任一类完全缺失、start/end 数量为奇数。

所有转换选择、跳过原因、逐 episode 原始区间、memory schema、30→15 Hz 设置和命令行都会
写入 LeRobot `meta/`，训练时再快照到 checkpoint metadata，供部署恢复与审计。

## 11. LeRobot 存放位置

不创建额外的数据根目录。converter 未传 `--output-root` 时直接使用 LeRobot 的
`HF_LEROBOT_HOME`：

```text
/root/.cache/huggingface/lerobot
  -> /mnt/public3/xcj/cache/huggingface/lerobot
```

本数据集的 repo id 为 `drawer_sorting_x1pro_shared_memory_sm2sm_15hz`，因此正式路径是：

```text
/root/.cache/huggingface/lerobot/drawer_sorting_x1pro_shared_memory_sm2sm_15hz
```

其真实共享存储位置自然落在上述软链目标内，与 RMBench 的其他 LeRobot 数据集完全一致。
