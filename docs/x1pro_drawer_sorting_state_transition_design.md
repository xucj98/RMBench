# X1Pro Drawer Sorting 状态转移与逐时间戳样本设计

## 1. 当前设计（v2）

本设计用于 `/mnt/public3/datasets/x1pro/table_clean` 中 `anno/tags.json` 含
`drawer_sorting` 的真机数据。原始视频和轨迹为 30 Hz，LeRobot 数据降采样到 15 Hz，
模型预测 30-step action chunk（2 秒）。机器人输入输出改为 S2M：

```text
state  = slave/follow state, 14D
action = master target,       14D
state_history_size = 0
state_future_size  = 0
latency_step       = 0
```

不再构造 state history/future，也不再使用 inpainting availability mask。action chunk
仍采用 soft boundary，可以跨状态边界连续执行。

旧版仅含 `drawer_target` 的 SM2SM 数据和训练配置保留为 v1，用于复现实验；它不再是当前
推荐设计。

## 2. 为什么需要两个 memory 字段

标签 4/5/6 的执行顺序不是固定的。119 条有效 episode 中六种排列均存在，因此
`item_1/2/3` 只能表示正在处理哪类物品，不能表示当前是第几层。

v2 使用两个正交字段：

```text
completed_layers ∈ {completed_0, completed_1, completed_2, completed_3}
drawer_target    ∈ {observe, item_1, item_2, item_3}
```

- `completed_layers` 表示按时间顺序已经完成的抓放次数，与 item id 无关；
- `drawer_target=observe` 表示应观察下一层；`item_i` 表示已经知道下一步应处理第 i 类物品；
- `(completed_3, observe)` 是任务完成状态，不再增加冗余的 phase/done 字段。

典型联合状态序列为：

```text
(0, observe) -> (0, item_i) -> (1, observe)
             -> (1, item_j) -> (2, observe)
             -> (2, item_k) -> (3, observe=done)
```

## 3. 标注和合法转移

`anno/sort.json` 的区间解释为 30 Hz 原始时间轴上的半开区间 `[start, end)`：

| 标注 | 事件 | drawer_target |
| --- | --- | --- |
| 1 / 2 / 3 | 观察到第一/二/三类物品 | `item_1/2/3` |
| 4 / 5 / 6 | 抓放第一/二/三类物品 | `item_1/2/3` |

字段各自的合法转移为：

```text
completed_0 -> {completed_0, completed_1}
completed_1 -> {completed_1, completed_2}
completed_2 -> {completed_2, completed_3}
completed_3 -> {completed_3}

observe -> {observe, item_1, item_2, item_3}
item_i  -> {item_i, observe}
```

模型使用 factorized fields，但两个字段属于同一个联合 memory。部署时除逐字段 transition
mask 外，还应以 `(completed_3, observe)` 判断完成。

## 4. 30 Hz ground-truth 时间轴

记原始 frame 为 `r`。先构造当前画面对应的输出 memory `m_plus[r]`。

### 4.1 drawer_target

1. episode 初始为 `observe`；
2. 进入 1/2/3 观察区间时变为对应 `item_i`，观察区间结束后继续锁存；
3. 进入 4/5/6 执行区间时再次强制为对应 `item_i`；
4. 到达执行区间的 `end` 时返回 `observe`。

第 3 条只是在已经开始执行的 timestamp 上补全动作条件。它不用于学习
`observe -> item_i`：执行区间内输入和输出都会被强制为同一个 `item_i`；若 item 仅由该
规则补全，Full State 的该字段 action loss 会被屏蔽。

### 4.2 completed_layers

将三个执行区间按 `start` 排序。第 `q` 个执行区间（`q=0,1,2`）内仍为
`completed_q`，只有到达该区间的 `end` 才更新为 `completed_(q+1)`：

```text
r < first_execution.end   : completed_0
r >= first_execution.end  : completed_1
r >= second_execution.end : completed_2
r >= third_execution.end  : completed_3
```

因此模型不会在“开始做第 q 层”时提前认为该层已完成。进度转移只从 4/5/6 的结束边界
学习，不从 1/2/3 观察区间学习。

## 5. 15 Hz 逐 timestamp 样本

转换后 timestamp 为 `k`，对应原始索引：

```text
r_k = clip(round(k * 30 / 15), 0, N - 1)
```

视频、机器人轨迹和 memory 都使用同一个 `r_k`。定义：

```text
m_plus[k]  = m_plus_30hz[r_k]
m_minus[k] = m_plus[max(k - 15, 0)]
```

15-frame lag 只用于构造 Serial 的 previous memory 以及 Full State 的 memory input；它不
代表 state history。若 `r_k` 位于按时间排序的第 `q` 个 4/5/6 执行区间，执行 override 为：

```text
m_minus.completed_layers = m_plus.completed_layers = completed_q
m_minus.drawer_target     = m_plus.drawer_target     = item_i
```

这保证即使缺少 1/2/3 观察片段，4/5/6 内的动作仍然被解释为“在已知第 q 层、已知 item_i
条件下的行为”，但不会凭执行画面伪造观察获得 item 的转移。

执行结束的第一个 sample 则自然形成：

```text
input  = (completed_q, item_i)       # 若 15 frame 前还在执行
target = (completed_(q+1), observe)  # 当前画面已完成
```

## 6. S2M robot 和 memory 布局

每个 LeRobot row `k` 使用：

```text
robot_state[k]  = slave[r_k]
robot_action[k] = master[r_(k+1)]
```

最后一个视频 frame 不生成 row。数据向量固定 padding 到 32D，以兼容现有 Pi0.5 接口。

两个字段都用 implicit-zero one-hot：类别 0 是全零，类别 1/2/3 分别是三维单位向量。

```text
state[0:14]   = slave state
state[14:17]  = encode(m_minus.completed_layers)
state[17:20]  = encode(m_minus.drawer_target)
state[20:32]  = 0

actions[0:14]  = next-frame master target
actions[14:17] = encode(m_plus.completed_layers)
actions[17:20] = encode(m_plus.drawer_target)
actions[20:32] = 0
```

### Full State

Full State adapter 读取上面的 dense memory。连续动作输出为 20D，再由模型 transform pad 到
32D。`memory_action_valid` 是两个字段各自的 mask；execution override 只会屏蔽没有真实
观察依据的 `drawer_target`，`completed_layers` 仍由执行结束标注直接监督。

### Serial Soft

Serial adapter 只读取 14D robot state/action，memory 使用同一数据集中的离散 sidecar：

```text
key_state_input_ids  = [completed_layers_minus, drawer_target_minus]
key_state_target_ids = [completed_layers_plus,  drawer_target_plus]
key_state_target_mask = [true, true]
```

训练使用 teacher forcing：先预测两个目标 state token，再把 GT token embedding 作为 action
预测条件。推理使用合法转移 mask 后的分类结果作为 action 条件和下一次 query 的 previous
memory。Full State 与 Serial Soft 因而共享同一个 LeRobot 数据集和同一个语义 schema，只
在表示 adapter 与模型路径上不同。

## 7. 跨边界 action loss

soft boundary 不截断 30-step chunk。为了避免 query 时不可知的未来 item 泄漏：

- query 已位于强制执行区间时，当前执行段的 robot action 可训练，因为输入已知 item；
- query 位于执行区间外，而 horizon 后面首次跨入一个仅靠 execution override 获得 item 的
  区间时，从该边界开始屏蔽剩余 robot action；
- Full State 对被强制获得的 `drawer_target` dense action 单独屏蔽；
- `completed_layers` 的转移及执行完成后的 `drawer_target -> observe` 保留监督。

## 8. 数据审计和存放位置

当前审计得到 119 条有效 episode，15 条跳过。每条有效 episode 必须恰好包含三个执行区间，
且 4/5/6 各一次；观察区间允许缺失。执行顺序按时间确定 layer rank。

converter 未指定 `--output-root` 时使用：

```text
/root/.cache/huggingface/lerobot
  -> /mnt/public3/xcj/cache/huggingface/lerobot
```

v2 数据集为：

```text
/root/.cache/huggingface/lerobot/drawer_sorting_x1pro_shared_memory_s2m_15hz_v2
```

转换配置、命令、逐 episode 区间、跳过原因、dense layout 与 schema 都写入数据集 `meta/`。
训练配置还会把同一 schema 写入 checkpoint policy metadata，供部署恢复与审计。

## 9. v1 兼容性

旧数据集 `drawer_sorting_x1pro_shared_memory_sm2sm_15hz`、旧训练配置
`pi05_x1pro_drawer_sorting_full_state` 和 `pi05_x1pro_drawer_sorting_serial_soft` 保留不变。
v2 使用新配置名：

```text
pi05_x1pro_drawer_sorting_s2m_full_state
pi05_x1pro_drawer_sorting_s2m_serial_soft
```

因此旧 checkpoint 仍能由旧配置加载；新实验不会静默改变旧实验的输入输出合同。
