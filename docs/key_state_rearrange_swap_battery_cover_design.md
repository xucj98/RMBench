# Rearrange / Swap / Battery / Cover Key-State 设计

本文先整理 4 个任务的状态设计，不直接规定最终训练配置。核心原则沿用 `put_back_block` 的经验：key state 至少分成两类。

```text
phase:
  严格随任务进度单调转移的状态机。phase 边界应来自 demo 的 move boundary 或显式记录的 phase boundary。

attribute:
  每条 episode 的关键属性。attribute 不一定在 phase 边界上变成可推断；
  每次属性更新都用 from_value -> to_value + update_window 表达，窗口外保持锁存。
```

不要把所有属性都按“phase 0 unknown，phase 1 known”处理。属性是否应更新只由它自己的 `update_window` 决定。

当前代码依据：

```text
envs/rearrange_blocks.py
envs/swap_blocks.py
envs/battery_try.py
envs/cover_blocks.py
description/task_instruction/rearrange_blocks.json
description/task_instruction/swap_blocks.json
description/task_instruction/battery_try.json
description/task_instruction/cover_blocks.json
```

注意：这 4 个任务当前 `demo_clean/scene_info.json` 的 `info` 基本为空，不像 `put_back_block` 已经写入 `origin_mat_id`。后续如果要稳定生成 key-state 训练数据，应该在重新生成数据时把下面列出的 task-specific metadata 写入 `scene_info.json`，不要依赖脆弱的字符串解析。

另一个需要注意的历史问题是：当前 `language_annotation.json` 的粒度并不完全统一。有些任务保留了多条低层 move annotation，有些任务更接近 high-level action annotation。后续 converter 不应假设“第几个 annotation 就是第几个小阶段”；正式数据应显式记录 `micro_stage_frame_ranges`。

## 通用标注规则

phase 和 attribute 不是同一种监督语义。

phase 暂时沿用 per-step target，便于复用当前 LeRobot action 序列：

```text
observation.state[f, phase_dims] = phase_input[f]
action[f, phase_dims]            = phase_gt[f + 1]
```

这个简化有缺陷：如果某个阶段动作失败，单纯按时间推进 phase 会过早进入下一个阶段。更合理的方案是让 phase 也在 query/chunk 级别由视觉确认后再更新。当前先保持 per-step phase，后续再做更稳的视觉确认式 phase update。

attribute 应按 query/chunk 级别理解：一次 policy query 根据当前观察判断 memory 是否需要更新。每次属性更新统一表示为：

```text
from_value: update_window 前已经锁存的值
to_value:   当前窗口内应输出并写入 memory 的新值
update_window: [start, end)
```

逐 query 规则：

```text
if f < start:
  attr_input[f]  = from_value
  attr_target[f] = from_value

if start <= f < end:
  attr_input[f]  = from_value
  attr_target[f] = to_value

if f >= end:
  attr_input[f]  = to_value
  attr_target[f] = to_value
```

其中 `unknown` 只是一个合法的初始 `from_value`，不是单独的规则分支。窗口内 input 不清空，仍然是窗口前的锁存值；只有 output target 变成新值。这样多次变化的属性不会在新窗口内忘记之前做过什么。

如果未来某个属性会在同一条 episode 中变化，就写成多个有序片段：

```text
[
  {from_value: unknown, to_value: 00, update_window: [start_00, end_00)},
  {from_value: 00,      to_value: 10, update_window: [start_10, end_10)},
  ...
]
```

片段之间不应互相重叠，且后一段的 `from_value` 应等于前一段的 `to_value`。第一段开始前 input/target 都是初始 `from_value`；进入新片段后，input 仍是 `from_value`，target 变成 `to_value`；窗口结束后 input/target 都锁存为 `to_value`。

实现上，如果模型仍通过 action chunk 输出 attribute 维度，同一个 query 产生的 chunk 内 attribute target 应保持同一个 `attr_target[f]`，而不是按未来每一帧展开成一条 attribute trajectory。也就是说，attribute target 是“这次观察后应写入的 memory”，不是“未来每一步的属性真值”。如果只靠 per-frame `action[f]` 字段无法表达这一点，需要在取出 action chunk 后由 transform/collator 覆盖 attribute dims。

编码上，V1 继续复用 pi0 当前的 32 维 `observation.state` 和 `action`。前 14 维保持机器人本体状态 / 动作不变，后 18 维按每个任务自己的 phase / attribute schema 解释。所有离散状态都用 one-hot 编码，不用 scalar id；phase 没有 `unknown` 类别，attribute 如果有未知状态，则把 `unknown` 作为该 attribute 的第 0 个 one-hot 类别。未使用的后续维度恒为 0。每个任务的数据集和 train config 都必须记录自己的 key-state layout；不要把不同任务的后 18 维直接混在一个多任务模型里解释。

## Rearrange Blocks

### 任务事实

桌面上有两张 mat，位置为 left / right；有两个红色方块：

```text
block1: 初始在两张 mat 中间。
block2: 初始在其中一张 mat 上。
empty mat: 另一张没有方块的 mat。
```

任务要求：

```text
1. 把中间的 block1 放到 empty mat 上。
2. 按一次按钮。
3. 把原来在 mat 上的 block2 放到两张 mat 中间。
```

环境成功判定要点：

```text
1. 按钮只能按一次；press_cnt > 1 直接失败。
2. 按钮按下时，block1 必须在原 empty mat 上，block2 必须仍在原 occupied mat 上。
3. 按钮复位后，block1 仍在原 empty mat 上，block2 在两张 mat 中间，右爪打开，则成功。

主要阈值：
  mat / middle xy 误差 < 0.03；
  block z < 0.77；
  button_joint < -0.005 视为按下，> -0.001 视为复位。
```

### Demo 小阶段

现有 `play_once` 可以拆成 11 个小阶段：

```text
s0  右臂抓取中间 block1。
s1  右臂抬起 block1。
s2  右臂把 block1 放到 empty mat。
s3  左臂接触按钮，同时右臂回初始位。
s4  左臂向下按按钮。
s5  左臂离开按钮。
s6  左臂回初始位。
s7  右臂移动到 block2 上方。
s8  右臂抓取 block2。
s9  右臂抬起 block2。
s10 右臂把 block2 放到两张 mat 中间。
```

### Phase 设计

建议 3 个 phase：

```text
0: move_middle_block_to_empty_mat   # s0..s2
1: press_button_after_first_move    # s3..s6
2: move_original_mat_block_to_middle # s7..s10
```

phase 严格按 demo 时间转移。评测时 phase 更新可以用模型输出，但 runtime update 规则应保持单调，只允许 `0 -> 1 -> 2`。

### Attribute 设计

需要的关键属性是 `empty_mat_side`：

```text
0: unknown
1: left
2: right
```

`block2` 初始所在 mat 是 `empty_mat_side` 的反面，因此不必再单独作为 policy state；metadata 中可以同时记录，方便检查。

属性更新窗口：

```text
transition:
  from_value: unknown
  to_value: empty_mat_side = left | right
  update_window: [episode_start, block1_place_start)

默认实现建议：
  如果有细粒度 move boundary，用 s0..s1 作为 acquisition window；
  即在 place segment 开始前完成 empty mat 推断。

临时 fallback：
  如果旧数据只有 high-level annotation，无法定位 place segment，可以先用整个
  move_middle_block_to_empty_mat 段生成调试数据，但不建议作为最终训练数据。
```

原因：初始 empty mat 在 `block1_place_start` 前直接可见；block1 放上去以后，“哪张 mat 最初是 empty”已经不能从当前单帧可靠恢复。因此窗口结束点应早于 block1 放置动作。

生成数据时建议写入 metadata：

```text
empty_mat_side: left | right
initial_occupied_mat_side: left | right
phase_sequence:
  - move_middle_block_to_empty_mat
  - press_button_after_first_move
  - move_original_mat_block_to_middle
micro_stage_sequence: s0..s10 对应的 frame ranges
```

### 编码方案

Rearrange Blocks 使用 6 个 key-state 维度，剩余 12 维 padding：

```text
dim 14:17  phase one-hot
           [move_middle_block_to_empty_mat,
            press_button_after_first_move,
            move_original_mat_block_to_middle]

dim 17:20  empty_mat_side one-hot
           [unknown, left, right]

dim 20:32  zero padding
```

## Swap Blocks

### 任务事实

桌面上有 3 个 tray，位置为 left / middle / right；两个红色方块分别在两个不同 tray 中，另一个 tray 为空。两个方块外观相同，但环境里 actor 身份不同。

任务要求：

```text
1. 每次只能移动一个方块。
2. 每个 tray 最多放一个方块。
3. 交换两个方块的 actor 位置。
4. 最后按按钮。
```

环境成功判定要点：

```text
1. block2 到达 block1 的初始 tray。
2. block1 到达 block2 的初始 tray。
3. 过程中不能把两个 block 放得太近或叠在一起，否则 fail flag 置位。
4. 右爪打开，按钮被按下，press_cnt == 1。

主要阈值：
  目标 tray x 误差 < 0.04；
  目标 tray y 误差 < 0.06；
  block z < 0.765；
  两个 block 的 x 误差 < 0.03 且 z 误差 < 0.01 时认为异常接近。
```

### Demo 小阶段

当前策略使用空 tray 做中转：

```text
s0  抓取 first block。
s1  抬起 first block。
s2  把 first block 放到 initial empty tray。
s3  操作臂回初始位。
s4  抓取 second block。
s5  抬起 second block。
s6  把 second block 放到 first block 的初始 tray。
s7  操作臂回初始位。
s8  从临时 tray 抓回 first block。
s9  抬起 first block。
s10 把 first block 放到 second block 的初始 tray。
s11 操作臂回初始位。
s12 左臂接触按钮。
s13 左臂向下按按钮。
```

### Phase 设计

建议 4 个 phase：

```text
0: move_first_block_to_empty_tray       # s0..s3
1: move_second_block_to_first_origin    # s4..s7
2: move_first_block_to_second_origin    # s8..s11
3: press_button                         # s12..s13
```

phase 严格单调转移，只允许 `0 -> 1 -> 2 -> 3`。

### Attribute 设计

这个任务的关键不是单纯记住 empty tray，而是记住 phase 0 已经做出的 swap plan commitment。两个方块外观相同，不能要求模型从图像里识别 simulator 内部的 `block1` / `block2` actor 身份；但在一次轨迹中，第一个被拿起的方块会自然成为 `first block`，它的来源 tray 就是后续阶段必须锁存的 `first_origin_tray`。

因此 `first_origin_tray` 不是一个视觉上唯一可推断的属性，而是一个 choice / plan commitment 属性：

```text
phase 0 开始时，两个 occupied tray 都是合法 first source。
训练标签跟随 expert trajectory 实际选择的 first source。
推理时，模型可以任选一个 occupied tray 作为 first source；一旦选择，就应把该选择锁存到后续 phase。
```

这保留了任务本身的多模态性，不需要把数据生成改成确定性 first-source 规则。Pi0 的 flow matching 训练可以处理“同一个可观察状态有多个合法动作”的分布；key state 这里只负责让后续阶段记住已经采样出的 plan。

建议 metadata 记录完整计划：

```text
initial_empty_tray: left | middle | right
first_origin_tray:  left | middle | right
second_origin_tray: left | middle | right
```

policy state 可以用两个字段表达最小充分信息：

```text
first_origin_tray:
  0 unknown, 1 left, 2 middle, 3 right

initial_empty_tray:
  0 unknown, 1 left, 2 middle, 3 right
```

`second_origin_tray` 可由剩余 tray 推出；如果后续为了 debug 和 overlay 更直观，也可以把三个字段都放进 state，但要保证三者满足 permutation 约束。

注意：这里的 `first_origin_tray` 与代码里的 `self.block1` 对应的是“本条 expert 轨迹第一个被搬走的物理方块来源”，不是要求模型识别红色方块的永久身份。phase 1 中当 second block 已经在空中时，视觉上会同时出现两个空 tray；没有 `first_origin_tray`，模型无法仅靠当前帧区分应把手中方块放回哪个空 tray。

属性更新窗口：

```text
transition:
  first_origin_tray:
    from_value: unknown
    to_value: left | middle | right
    update_window: [episode_start, first_block_place_start)

  initial_empty_tray:
    from_value: unknown
    to_value: left | middle | right
    update_window: [episode_start, first_block_place_start)

默认实现建议：
  用 s0..s1 作为 acquisition window；
  对 initial_empty_tray，这是视觉可推断窗口；
  对 first_origin_tray，这是 choice / plan commitment 窗口，不是唯一识别窗口；
  即在 first block 的 place segment 开始前完成选择并写入 memory。

临时 fallback：
  如果旧数据只有 high-level annotation，无法定位 place segment，可以先用
  move_first_block_to_empty_tray 整段生成调试数据，但不建议作为最终训练数据。
```

数据生成不应为了简化 key state 而把 `first_origin_tray` 改成确定性规则，否则会改变 expert 策略分布，也会让和原始 baseline 的对比不公平。需要做的是在 metadata 中记录当前轨迹实际选择的 `first_origin_tray`，用于训练 label 和 overlay 检查。

生成数据时建议写入 metadata：

```text
initial_empty_tray
first_origin_tray
second_origin_tray
tray_poses
phase_sequence:
  - move_first_block_to_empty_tray
  - move_second_block_to_first_origin
  - move_first_block_to_second_origin
  - press_button
micro_stage_sequence: s0..s13 对应的 frame ranges
```

### 编码方案

Swap Blocks 使用 12 个 key-state 维度，剩余 6 维 padding：

```text
dim 14:18  phase one-hot
           [move_first_block_to_empty_tray,
            move_second_block_to_first_origin,
            move_first_block_to_second_origin,
            press_button]

dim 18:22  initial_empty_tray one-hot
           [unknown, left, middle, right]

dim 22:26  first_origin_tray one-hot
           [unknown, left, middle, right]

dim 26:32  zero padding
```

`second_origin_tray` 不单独编码，由 three-tray permutation 推出：

```text
second_origin_tray = {left, middle, right} - initial_empty_tray - first_origin_tray
```

## Battery Try

### 任务事实

桌面上有左右两个 battery 和一个带 dashboard needle 的 battery slot。电池可以用两种方向放入 slot：

```text
0: positive direction
1: negative direction
```

环境固定尝试组合顺序：

```text
00 -> 10 -> 11 -> 01
```

其中 `00` 不是正确答案；`10`、`11`、`01` 中随机一个是正确组合。尝试到正确组合时，dashboard needle 变为 on，demo 停止。

任务成功判定要点：

```text
1. 两个 battery 都在 slot 的目标位置。
2. 当前组合等于 correct_combination。
3. dashboard needle 已经转到 on。

主要阈值：
  battery 到目标 slot 位置的 xy 距离 < 0.02；
  needle_joint > 1.4 视为 dashboard on。
```

### Demo 小阶段

demo 长度随正确组合变化：

```text
s0  双臂抓取两个 battery。
s1  双臂抬起两个 battery。
s2  放入左 battery，方向 0。
s3  左臂回初始位。
s4  放入右 battery，方向 0。
s5  右臂回初始位。

s6  如果 10 还未达成，抓取左 battery。
s7  抬起左 battery。
s8  放入左 battery，方向 1；检查 dashboard。
s9  左臂回初始位。

s10 如果 11 还未达成，抓取右 battery。
s11 抬起右 battery。
s12 放入右 battery，方向 1；检查 dashboard。
s13 右臂回初始位。

s14 如果 01 还未达成，抓取左 battery。
s15 抬起左 battery。
s16 放入左 battery，方向 0；检查 dashboard。
s17 左臂回初始位。
```

若 `10` 正确，demo 在 s9 后结束；若 `11` 正确，demo 在 s13 后结束；若 `01` 正确，demo 在 s17 后结束。

### Phase 设计

建议 4 个 phase：

```text
0: place_initial_00       # s0..s5
1: try_10                 # s6..s9
2: try_11                 # s10..s13
3: try_01                 # s14..s17
```

phase 是 trial progress，不是隐藏答案。它严格按尝试顺序前进；如果 dashboard on，episode 结束，不再进入后续 phase。

### Attribute 设计

这个任务正式 policy state 不需要额外 attribute。当前任务本质是固定尝试流程：

```text
00 -> 10 -> 11 -> 01
```

episode 会在正确组合处直接结束；评测循环在 `check_success()` 变 true 后也会终止当前 rollout。因此模型不需要读取 dashboard needle 后再决定是否继续执行后续尝试，也不需要锁存 `found_combo`。

不应把下面这些量放进 policy input 或 action target：

```text
correct_combination  # 尝试前不可观测，是隐藏答案
found_combo          # 只有成功后才知道，但成功后 episode 已结束
dashboard_on         # 成功判定 / debug 信息，不是后续决策所需状态
current_combo        # 已由 phase 表示，不必重复作为 attribute
```

如果后续做 overlay 或 debug，可以显示这些量，但它们应只来自 env metadata 或 success check，不作为 key-state 训练监督。

生成数据时建议写入 metadata：

```text
correct_combination: 10 | 11 | 01
trial_order: [00, 10, 11, 01]
actual_trial_sequence: 运行到成功为止的组合序列
dashboard_on_after_trial: 10 | 11 | 01
policy_state:
  phase only
phase_sequence:
  - place_initial_00
  - try_10
  - try_11
  - try_01
micro_stage_sequence: s0..s17 的实际 frame ranges
```

### 编码方案

Battery Try 使用 4 个 key-state 维度，剩余 14 维 padding：

```text
dim 14:18  phase one-hot
           [place_initial_00,
            try_10,
            try_11,
            try_01]

dim 18:32  zero padding
```

`correct_combination`、`dashboard_on`、`found_combo`、`current_combo` 都不进入 policy key-state 编码，只作为 metadata / debug / overlay 信息。

## Cover Blocks

### 任务事实

桌面上有 3 个 block，颜色为 red / green / blue，随机排列在 left / middle / right 三个位置；另有 3 个 cover，初始也在 left / middle / right 三个位置。

任务要求分两段：

```text
1. 按位置从 left -> middle -> right 依次盖住三个 block。
2. 再按颜色 red -> green -> blue 的顺序依次打开对应位置的 cover。
```

环境成功判定要点：

```text
1. cover mask 必须严格走目标序列，不能跳步。
2. 第一段固定为 000 -> 100 -> 110 -> 111。
3. 第二段根据 RGB 颜色所在位置，把对应位从 1 变成 0。
4. 任意非 tmp 状态不等于当前或下一目标状态，会置 fail_flag。
5. 指针到达 target_state_transition 最后一个状态且 fail_flag 为 False，则成功。

主要阈值：
  cover 与对应 block 的 xy 误差 < 0.03 且 cover z < 0.742 时认为盖住；
  每个 cover 必须在初始 open 位或对应 close 位附近，xy 误差 <= 0.035 且 z <= 0.742，
  否则当前 mask 记为 tmp，不推进成功状态。
```

### Demo 小阶段

建议按 6 个 high-level action 组织；每个 high-level 内部有抓取、抬起、放置、离开等小动作。跨手切换时的 `back_to_origin` 归入前一个 high-level phase。

```text
s0  cover_left_position:
    抓 left cover，抬起，放到 left block 上，离开；必要时回初始位。

s1  cover_middle_position:
    抓 middle cover，抬起，放到 middle block 上，离开；必要时回初始位。

s2  cover_right_position:
    抓 right cover，抬起，放到 right block 上，离开；必要时回初始位。

s3  uncover_red_block:
    根据 red block 的位置，打开对应位置的 cover。

s4  uncover_green_block:
    根据 green block 的位置，打开对应位置的 cover。

s5  uncover_blue_block:
    根据 blue block 的位置，打开对应位置的 cover。
```

### Phase 设计

建议 6 个 phase：

```text
0: cover_left_position
1: cover_middle_position
2: cover_right_position
3: uncover_red_block
4: uncover_green_block
5: uncover_blue_block
```

phase 严格按 `0 -> 1 -> 2 -> 3 -> 4 -> 5` 转移。`cover_mask` 是环境验证状态，可以用于日志和 overlay；V1 不建议单独作为 policy memory，因为 phase 已经表达了下一步语义目标。

### Attribute 设计

关键属性是颜色到位置的映射：

```text
red_pos:
  0 unknown, 1 left, 2 middle, 3 right

green_pos:
  0 unknown, 1 left, 2 middle, 3 right

blue_pos:
  0 unknown, 1 left, 2 middle, 3 right
```

metadata 中也可以存成更紧凑的：

```text
rgb_positions: [red_pos, green_pos, blue_pos]
```

其中三者必须是 left / middle / right 的一个 permutation。

属性更新窗口是本任务最关键的设计点：

```text
transition:
  red_pos:
    from_value: unknown
    to_value: left | middle | right
    update_window: [episode_start, first_cover_place_start)

  green_pos:
    from_value: unknown
    to_value: left | middle | right
    update_window: [episode_start, first_cover_place_start)

  blue_pos:
    from_value: unknown
    to_value: left | middle | right
    update_window: [episode_start, first_cover_place_start)
```

不能简单把整个 `cover_left_position` 都当作 acquisition window。原因是 `cover_left_position` 后半段中 left block 已经被 cover 遮住；如果此时 `rgb_positions_input` 仍是 unknown，模型就必须从已经不可见的颜色中预测完整 permutation，会引入错误监督。

实现建议：

```text
1. 生成新数据时，必须记录 micro-stage boundary，至少要能定位第一个 cover 的 place segment。
2. 默认 update_window 结束点设置为 first cover place segment 开始前。
3. update_window 结束后，red_pos / green_pos / blue_pos 全部进入 memory 并锁存。
```

可选消融：

```text
color_late_update_window:
  将 update_window 延长到 [episode_start, second_cover_place_start)。

理由：
  left block 被盖住后，middle / right block 仍可见；在已知颜色集合为 RGB 且三色各出现一次的前提下，
  left block 的颜色可由缺失颜色推出。

风险：
  这会要求模型在单帧中做集合排除推理，并依赖“没有重复颜色”的任务先验。
  默认方案不依赖这个推理，因此更稳。
```

如果只能拿到当前 `language_annotation.json` 的 high-level 段长，而没有 micro-stage boundary，则不建议直接生成最终训练数据；应先补充数据生成 metadata 或重新生成带边界的数据。

生成数据时建议写入 metadata：

```text
rgb_positions:
  red:   left | middle | right
  green: left | middle | right
  blue:  left | middle | right
open_order_positions: [red_pos, green_pos, blue_pos]
target_state_transition:
  - "000"
  - "100"
  - "110"
  - "111"
  - 后续按 red_pos / green_pos / blue_pos 逐位打开
phase_sequence:
  - cover_left_position
  - cover_middle_position
  - cover_right_position
  - uncover_red_block
  - uncover_green_block
  - uncover_blue_block
micro_stage_sequence: 每个 cover/open 内部动作及 frame ranges，尤其是 first cover place boundary
```

### 编码方案

Cover Blocks 使用 18 个 key-state 维度，刚好用满后 18 维：

```text
dim 14:20  phase one-hot
           [cover_left_position,
            cover_middle_position,
            cover_right_position,
            uncover_red_block,
            uncover_green_block,
            uncover_blue_block]

dim 20:24  red_pos one-hot
           [unknown, left, middle, right]

dim 24:28  green_pos one-hot
           [unknown, left, middle, right]

dim 28:32  blue_pos one-hot
           [unknown, left, middle, right]
```

`red_pos / green_pos / blue_pos` 是颜色到空间位置的映射，不表示 left / middle / right 的空间排列固定为 RGB。颜色命名按 `envs/cover_blocks.py` 中 `color_tuple = [(1,0,0), (0,1,0), (0,0,1)]` 和任务语言中的 red / green / blue 定义。

## 下一步实现任务

本轮目标是完成 `rearrange_blocks`、`swap_blocks`、`battery_try`、`cover_blocks` 四个任务的 pi0 + key-state 训练与评测，得到正式 SR。

本批次 batch id：

```text
pi0_key_state_baseline
```

这批实验定义为 pi0 key-state baseline 的整体批次；本文档先落地 `rearrange_blocks`、`swap_blocks`、`battery_try`、`cover_blocks` 四个任务，后续剩余任务和已有 `put_back_block` key-state baseline 结果也应归并到同一批次记录中。

执行方式不是等所有代码写完再一次提交，而是按可验收阶段推进。每个阶段都按下面顺序收尾：

```text
完成本阶段实现或运行 -> 对照本阶段验收标准自查 -> 提交 tracked 代码/配置；如果是正式运行阶段，再按下面规则更新 experiments/pi0_key_state_baseline/README.md
```

正式数据生成、训练和评测必须有同一入口的小规模 smoke 作为门禁：

```text
正式运行前：
  1. 用同一代码路径和同类参数跑 smoke。
  2. 对照本阶段 smoke/验收标准自查。
  3. 修复发现的问题；如有修改，重新 smoke。
  4. 提交 tracked 代码、配置和入口。
  5. 删除对应的 `_smoke` 产物，避免正式运行误用 smoke 数据、LeRobot repo、checkpoint 或 eval_result。
  6. 确认正式运行相关 worktree 干净。
  7. 在这个 clean commit 上启动正式数据生成、训练或评测。

正式运行后：
  1. 如果是训练、评测这类长时间任务，确认正式任务已经启动后，先更新并提交 `experiments/pi0_key_state_baseline/README.md`，写清楚批次状态、目标路径、wandb id 或运行状态。
  2. 如果是数据转换这类可能较短的任务，可以等完成后一次更新；如果预计运行很久，也按长时间任务先记录目标 repo_id、路径和运行状态。
  3. 正式任务完成后，对照本阶段正式验收标准检查产物和 metadata。
  4. 不提交数据、checkpoint、eval result、视频和大日志。
  5. 再次更新并提交 `experiments/pi0_key_state_baseline/README.md`，补充结果、完成状态、路径、wandb id 和必要结论。
```

smoke test 可以在未提交工作区运行。smoke 产物应集中放在容易统一清理的位置，并使用 `_smoke` 或 `data_smoke/` 这类明确名称。LeRobot repo、checkpoint、eval_result 这类 smoke 产物放在对应正式产物的同一父目录下；数据采集 smoke 由于现有路径规则使用 `data_smoke/<task>/demo_clean_state`；norm stats smoke 是例外，直接写正式 `norm_stats.json` 路径，正式训练前重新计算并覆盖。正式运行前删除对应 smoke 产物。smoke 不能进入 `experiments/pi0_key_state_baseline/README.md`。数据、checkpoint、eval result、视频和大日志不进 git；正式阶段启动和完成时才更新并提交 `experiments/pi0_key_state_baseline/README.md`。

`experiments/pi0_key_state_baseline/README.md` 是批次索引和结果摘要，不手写启动时间、git commit、完整命令或环境变量。正式数据、LeRobot repo、checkpoint 和 eval_result 必须由代码在各自产物目录下自动保存 `command.txt`、resolved config 和必要 metadata；复现时以这些自动记录为准。README 只记录实验目的、任务列表、数据/ckpt/eval/wandb 的定位信息、验收摘要、SR 和结论。

当前 `task_config/` 整体被 `.gitignore` 忽略，只有已经进入 git 的少量基础配置可作为正式依赖。本轮应新增一个正式的 `task_config/demo_clean_state.yml`，并用 `git add -f` 纳入 git 管理；正式命令不能依赖本机临时的 `task_config/demo_clean_state_*.yml`。

### 1. 数据采集入口、task config 与记录机制

先改采集入口，不生成正式 50ep 数据。目标是让 `script/collect_data.py` 和 instruction 生成脚本支持 CLI override，并把 override 后的有效配置保存下来。

正式数据使用 tracked `task_config/demo_clean_state.yml`。该文件应基本继承 `demo_clean.yml` 的语义，保持 `save_path: ./data`、`episode_num: 50`、`use_seed: false` 等正式设置。这样可以复用现有路径规则：

```text
final_data_dir = <save_path>/<task_name>/<task_config>
```

即下面的命令会写到 `data/<task>/demo_clean_state/`：

```bash
python script/collect_data.py <task> demo_clean_state
```

CLI override 仍然需要支持，但只用于 smoke 或临时覆盖，例如把 `episode_num` 改成 1，或把 `save_path` 改成 `./data_smoke`。它不承担定义正式数据 setting 的职责。

正式输出目录结构：

```text
data/<task>/demo_clean_state/
  data/
  instructions/
  scene_info.json
  language_annotation.json
  seed.txt
  _traj_data/
  metadata/
    command.txt
    config.yaml
```

`description/gen_episode_instructions.sh` 和 `description/utils/generate_episode_instructions.py` 也应读取同一个 tracked config，并使用相同的 resolved config 规则定位 `scene_info.json` 和 `instructions/`。正式路径仍是 `data/<task>/demo_clean_state`；数据采集 smoke 路径使用 `data_smoke/<task>/demo_clean_state`，正式数据生成前删除。

验收标准：

```text
1. collect_data 和 instruction 生成脚本都支持 CLI overrides。
2. `task_config=demo_clean_state` 是 tracked config，正式输出自然落到 data/<task>/demo_clean_state。
3. metadata/command.txt 由代码直接生成，记录 git commit、cwd、白名单 env 和完整启动命令。
4. metadata/config.yaml 由代码直接生成，保存合并 overrides 后的 resolved config。
5. command.txt 中的正式命令不引用 ignored 的 task_config/demo_clean_state_*.yml。
6. 不修改或覆盖 data/<task>/demo_clean。
7. 本阶段只跑 CLI/路径/metadata 的轻量 smoke，不生成正式 50ep 数据。
```

提交要求：

```text
提交 collect_data、instruction 生成入口、task_config/demo_clean_state.yml。
不提交任何 smoke 数据。
```

### 2. Env Metadata

给四个 env 增加结构化 metadata 写入。env 只记录 episode 事实和真实执行边界，不记录 one-hot layout、margin、jitter、训练配置等 converter/实验配置。

`scene_info["info"]` 至少包含：

```text
task_facts:
  当前任务需要的客观属性，例如 empty_mat_side、initial_empty_tray、
  first_origin_tray、rgb_positions、trial_order 等。

micro_stages:
  每个关键低层动作片段的 name、start_frame、end_frame。
```

验收可以先人工检查，不强制新增独立 validator 脚本。1ep smoke 时直接查看 `scene_info.json`、`language_annotation.json`、`instructions/episode0.json` 和 hdf5 frame 数即可。正式 50ep 数据生成后，可以用临时检查命令或简短脚本批量统计，但这不是必须进入 git 的阶段交付物。

验收标准：

```text
1. rearrange_blocks 记录 empty_mat_side 和三段 high-level phase 所需边界。
2. swap_blocks 记录 initial_empty_tray、first_origin_tray，并保留 expert trajectory 的多模态选择，不改成确定性规则。
3. battery_try 记录 trial_order、actual_trial_sequence、correct_combination；policy state 只使用 phase。
4. cover_blocks 记录 red_pos、green_pos、blue_pos，并能定位 first_cover_place_start。
5. 每个 episode 的 micro_stages frame range 非空、递增，并和 hdf5 action frame 数可对齐。
6. 四个任务各跑 1ep smoke，生成 hdf5、scene_info.json、language_annotation.json、instructions/、metadata/command.txt、metadata/config.yaml。
7. 人工检查四个 1ep smoke 数据：
   - 必要文件存在。
   - scene_info 中每个 episode 都有 info.task_facts 和 info.micro_stages。
   - task_facts 包含当前任务必需字段。
   - micro_stages 的 start_frame/end_frame 为整数、递增、非空，并落在 hdf5 action frame 范围内。
   - language_annotation 和 instructions episode 数与 hdf5 episode 数一致。
8. smoke 输出放在 data_smoke/<task>/demo_clean_state；正式数据生成前删除，不进入正式数据目录，也不写入 `experiments/pi0_key_state_baseline/README.md`。
```

提交要求：

```text
提交四个 env metadata 实现。
不提交 smoke 数据。
```

### 3. 正式 Demo 数据生成

第 2 阶段的 1ep smoke 通过并提交后，在 clean commit 上生成四个任务的正式 `demo_clean_state` 数据。正式生成前目标目录必须为空或被明确清理，不能混入 smoke 或半成品 episode。

正式命令形态：

```bash
python script/collect_data.py <task> demo_clean_state
```

验收标准：

```text
1. 四个任务各有 50 个 hdf5 episode、50 个 instruction json、seed.txt、_traj_data/、scene_info.json、language_annotation.json。
2. 每个 episode 的 scene_info.info 都包含 task_facts 和 micro_stages。
3. 每个 demo_clean_state/metadata/ 下都有 command.txt 和 resolved config.yaml。
4. instructions/ 正常生成，并写入 demo_clean_state 目录。
5. command.txt 中的 commit 是第 2 阶段提交后的 clean commit。
6. 对四个正式数据目录做批量验收，确认必要文件、episode 数、task_facts、micro_stages、instruction 数和 hdf5 frame 对齐；验收结果写入 `experiments/pi0_key_state_baseline/README.md`。
```

提交要求：

```text
不提交 data/<task>/demo_clean_state 大文件。
提交 `experiments/pi0_key_state_baseline/README.md`，写清楚 data path 和校验结果。
正式数据生成完成后必须先按验收标准自查，再提交 README 更新。
```

如果后续 converter 阶段发现 metadata schema 设计错误，应回到第 2 阶段修改 env metadata，重新 smoke、提交，并重新生成正式数据；不要在已经生成的数据上手工打补丁。

### 4. 通用 Converter 与 Config

新增或重构通用 converter，使其从 tracked converter config 读取任务定义，把 RMBench hdf5 数据转成 LeRobot 数据集。converter 负责所有 key-state 数据处理，包括 phase 合并、attribute 更新窗口、one-hot 编码和 padding。

每个任务应有一个被 git 管理的 converter config。推荐放在实验批次目录下：

```text
experiments/pi0_key_state_baseline/converter_configs/
  rearrange_blocks.yaml
  swap_blocks.yaml
  battery_try.yaml
  cover_blocks.yaml
```

本轮先保持每个任务一份显式 config，便于审阅和复现。config 不需要极简；凡是会影响转换结果、prompt 选择、输出 repo 或后续追溯的信息，都可以写进去。

一个样例：

```yaml
dataset:
  task: swap_blocks
  source_dir: data/swap_blocks/demo_clean_state
  repo_id: swap_blocks_demo_clean_state_key_state
  episodes: 50

state_layout:
  state_dim: 32
  robot_dim: 14
  padding: zero

phase:
  dim: [14, 17]
  encoding: one_hot
  labels:
    - move_first_to_empty
    - move_second_to_first_origin
    - move_first_to_second_origin
  ranges:
    move_first_to_empty: [episode_start, micro_stages.first_place.end_frame]
    move_second_to_first_origin: [micro_stages.first_place.end_frame, micro_stages.second_place.end_frame]
    move_first_to_second_origin: [micro_stages.second_place.end_frame, episode_end]

attributes:
  - name: initial_empty_tray
    dim: [17, 21]
    encoding: one_hot
    labels: [unknown, left, middle, right]
    transitions:
      - from_value: unknown
        to_value: task_facts.initial_empty_tray
        update_window: [episode_start, micro_stages.first_pick.end_frame]
  - name: first_origin_tray
    dim: [21, 25]
    encoding: one_hot
    labels: [unknown, left, middle, right]
    transitions:
      - from_value: unknown
        to_value: task_facts.first_origin_tray
        update_window: [episode_start, micro_stages.first_pick.end_frame]

metadata:
  copy_source_data_metadata: true
```

config 中的引用不再额外定义 `time_refs`。converter 只需要支持少量固定解析规则：

```text
episode_start:
  当前 episode 的第 0 个 action frame。

episode_end:
  当前 episode 的 action frame 数。

task_facts.<field>:
  从 scene_info["episode_i"]["info"]["task_facts"][field] 取值。可用于 from_value、to_value 或其它值字段。

micro_stages.<stage_name>.start_frame / end_frame:
  从 scene_info["episode_i"]["info"]["micro_stages"] 中按 name 找到对应 stage，再取 start_frame 或 end_frame。
```

因此 `micro_stages` 的 `name` 必须是稳定、可被 config 引用的标识符。converter 不应解析 `first_place_end` 这类没有路径来源的裸符号。

正式 config 写 `dataset.episodes: 50`。smoke conversion 用 CLI override 改成 `dataset.episodes=1`，正式转换则直接使用 config 中的 episode 数。converter 应校验 `source_dir/data/` 中实际 hdf5 episode 数不少于 `dataset.episodes`；实际转换的 episode 范围和数量应写入 LeRobot repo 的 `meta/rmbench/key_state_config.yaml` resolved config 中。`instruction_type` 默认使用 `seen`，因为当前正式训练使用 seen instruction；如果后续需要用 `unseen`，也用 CLI override，而不是每个任务 config 重复写。

这里的 `from_value -> to_value` 是 attribute 的状态转移语义。`from_value` 和 `to_value` 使用同一套解析规则：如果字符串形如 `task_facts.xxx`，就从 `scene_info.info.task_facts` 取值；否则按 attribute labels 中的固定枚举值解释，例如 `unknown`、`left`、`middle`、`right`。

phase 是 per-step label；attribute 是可在一个时间窗口内被推断并在窗口后锁存的值。当前四个任务的 attribute 都是一次转移，因此 converter 可以生成：

```text
attribute state:
  update_window 内为 from_value，窗口后为 to_value

attribute action:
  update_window 内为 to_value，窗口后为 to_value
```

后续如果出现 `unknown -> A -> B` 这类多阶段 attribute，可以在同一个 attribute 下增加多条 `transitions`。本轮实现可以先只支持单条 transition；如果 config 出现多条 transition 而代码尚未支持，converter 必须报错，不能静默生成错误标签。

小规模 smoke conversion 使用同一个 converter 入口，但用 CLI 限制 episode 数，并通过 override 覆盖 `dataset.repo_id` 写到同一 LeRobot 根目录下的 `_smoke` repo，例如：

```bash
bash experiments/pi0_key_state_baseline/commands/convert_smoke.sh
```

`experiments/pi0_key_state_baseline/commands/convert_smoke.sh` 内部逐个调用 converter，例如：

```bash
python policy/pi05/examples/aloha_real/convert_robotwin_key_state_to_lerobot.py \
  --config experiments/pi0_key_state_baseline/converter_configs/swap_blocks.yaml \
  --overrides \
    dataset.episodes=1 \
    dataset.repo_id=swap_blocks_demo_clean_state_key_state_smoke
```

converter 需要在 LeRobot repo 下写入 RMBench 自己新增的复现 metadata。LeRobot 标准 `meta/` 文件（如 `info.json`、`episodes.jsonl`、`episodes_stats.jsonl`、`tasks.jsonl`）只服务数据集加载和统计，不作为本实验复现快照的核心内容。

```text
meta/rmbench/key_state_config.yaml
meta/rmbench/convert_command.txt
meta/rmbench/source_data_config.yaml
meta/rmbench/source_data_command.txt
```

其中 `key_state_config.yaml` 是 converter 的 resolved config；`convert_command.txt` 是 converter 启动命令和 commit；`source_data_config.yaml` 和 `source_data_command.txt` 分别复制自 `source_dir/metadata/config.yaml` 和 `source_dir/metadata/command.txt`，用于追溯 demo 数据本身如何生成。

验收标准：

```text
1. 同一个 converter 能通过不同 converter config 转换四个任务。
2. 生成的 observation.state/action 都是 32 维，前 14 维为 robot，后 18 维按各任务 config 解释。
3. phase one-hot、attribute one-hot、padding zero 的校验全部通过。
4. converter 不依赖 language_annotation 的“第几个文本片段”来推断关键边界；关键边界来自 scene_info.info.micro_stages。
5. 四个任务各有 tracked converter config，并且 smoke conversion 使用这些 config。
6. meta/rmbench/key_state_config.yaml 是 resolved config，meta/rmbench/convert_command.txt 记录 commit、cwd、白名单 env 和命令。
7. meta/rmbench/source_data_config.yaml 和 meta/rmbench/source_data_command.txt 与 source_dir/metadata 下的文件一致。
8. 本阶段先用正式 demo 数据做小规模 smoke conversion，例如每个任务转换 1ep 或少量 episode。
```

提交要求：

```text
提交 converter、四个 converter config 和 `experiments/pi0_key_state_baseline/commands/convert_smoke.sh`。
不提交 LeRobot smoke 数据。
```

### 5. 正式 LeRobot 转换

第 4 阶段的小规模 conversion smoke 通过并提交后，在 clean commit 上把四个 `demo_clean_state` 转成正式 LeRobot 数据集。本阶段只负责生成 LeRobot repo；不要在 train config 尚未确定前计算 norm stats。

验收标准：

```text
1. 每个任务都有明确的 repo_id，repo_id 表示数据集内容。
2. 每个 LeRobot repo 都包含 meta/rmbench/key_state_config.yaml、meta/rmbench/convert_command.txt、meta/rmbench/source_data_config.yaml、meta/rmbench/source_data_command.txt。
3. 每个 LeRobot repo 的 episode 数、state/action 维度和 key-state one-hot 校验通过。
4. repo_id、LeRobot repo 路径和校验结果记录在 `experiments/pi0_key_state_baseline/README.md` 中；转换命令和 commit 以 meta/rmbench/convert_command.txt 为准。
```

提交要求：

```text
不提交 LeRobot 数据集。
如果正式转换预计一两个小时内完成，可以完成后一次性更新并提交 `experiments/pi0_key_state_baseline/README.md`，写清楚每个任务的 repo_id、LeRobot repo 路径和校验结果。
如果正式转换预计运行很久，应在确认启动后先更新并提交 README，记录目标 repo_id、目标路径和运行状态；完成后再按验收标准自查并补充校验结果。
```

### 6. pi05 Key-State 训练入口、Norm Stats 与训练 Smoke

第 5 阶段 README 更新提交后，实现 pi05 训练和评测入口，计算或校验 pi05 训练需要的 norm stats，并跑最小训练 smoke。train config、metadata 保存、norm stats 和训练 smoke 是一个闭环，不拆成独立阶段。

训练侧 `KeyStateAlohaInputs/Outputs` 已经能保留 14 维之后的 key-state 维度；需要重点确认 train config、deploy policy、eval wrapper 和 overlay 都不再硬编码 put_back_block 的 8 维 layout，而是读取当前 run 的 key-state config/layout。

训练配置遵循 pi05 两层 checkpoint 结构：

```text
policy/pi05/checkpoints/<train_config_name>/<exp_name>/<step>
```

本批四个任务应共用一个 `train_config_name`，例如：

```text
pi0_aloha_key_state_lora
```

四个任务的差异是 run-level 差异，不应拆成四个 train config。训练入口必须支持用 CLI override 或等价的 batch runner 参数覆盖：

```text
exp_name
data.repo_id
```

不要向 `policy_metadata` 写入 `key_state_schema`、`key_state_variant`、`batch_id`、`task` 等字段。`policy_metadata` 是 OpenPI 原本给 policy server 的元信息通道；本批实验保持它为空，或只保留上游 pi0 原本已有且运行时确实需要的字段，不承担复现记录职责。

训练阶段不应重新解析数据生成和 converter 的所有历史信息。它应通过 `data.repo_id` 定位 LeRobot repo，并把该 repo 的 `meta/rmbench/` 四文件快照复制到 checkpoint 目录，例如：

```text
policy/pi05/checkpoints/<train_config_name>/<exp_name>/
  metadata/
    train_config.yaml
    command.txt
    rmbench_data_meta/
      key_state_config.yaml
      convert_command.txt
      source_data_config.yaml
      source_data_command.txt
```

只有模型结构、LoRA/full finetune 范式、数据 transform、归一化规则或训练范式发生变化时，才新增 `train_config_name`。单纯 task、repo_id、exp_name、seed、GPU 或 key-state config 路径不同，不应新增 `train_config_name`。

key-state config/layout 不是给模型网络本身看的训练超参。训练时模型只看到 32 维 `observation.state` 和 `action`；key-state config 的作用是运行时和追溯：

```text
1. 说明 state[14:32] 中哪些 slice 是 phase / attribute。
2. 说明每个 slice 的 labels 和 encoding，便于初始化 memory。
3. 说明 eval 时如何从模型 action 的 key-state 维度更新下一次 query 的 memory。
4. 说明 overlay 如何显示当前 phase / attribute。
5. 记录本 run 使用的 converter config 和 source data config，便于 checkpoint、wandb 和 eval_result 追溯。
```

因此训练入口不需要单独维护一份 `key_state_schema` 参数，也不需要通过 `policy_metadata` 传 `key_state_schema` 或 `key_state_variant`。训练时把 resolved train config、启动命令和 LeRobot repo 的 `meta/rmbench/` 四文件快照保存进 checkpoint；eval wrapper 和 overlay 从 checkpoint metadata 读取 key-state config/layout。否则如果四个任务共用同一个 `train_config_name`，eval 只看静态 config 会丢失每个 task 的 layout。

wandb 上也按同一原则保存：训练脚本会把 resolved `TrainConfig` 写入 `wandb.config`，但不通过 `policy_metadata` 上传 key-state schema 或实验 metadata。完整复现 metadata 作为 run file 上传，内容仅包含 `meta/rmbench/` 的四个文件。不要把 LeRobot 标准 `meta/` 整目录上传为本实验的复现快照。

norm stats 必须使用同一个 `train_config_name` 和对应任务的 `data.repo_id` override，不能在 train config 尚未确定时提前计算。`scripts/compute_norm_stats.py` 已支持 `--max-frames`，因此本阶段先用小 `max_frames` 做 norm stats smoke，验证 repo_id、transform、assets 路径和写入权限；smoke 会写到正式 norm stats 路径。第 7 阶段正式训练前再不带 `--max-frames` 重新计算并覆盖同一个文件。

```text
policy/pi05/assets/pi0_aloha_key_state_lora/<repo_id>/norm_stats.json
```

训练 smoke 使用和正式训练相同的 `train_config_name` 和 `data.repo_id`，但用单独的 smoke `exp_name` 隔离 checkpoint，例如：

```text
<task_name>_smoke
```

验收标准：

```text
1. 四个任务共用同一个 train_config_name。
2. 四个任务通过 override 提供各自的 data.repo_id 和 exp_name；不通过 policy_metadata 传 key-state schema 或实验 metadata。
3. 每个任务先完成 `--max-frames` norm stats smoke，并写入 policy/pi05/assets/pi0_aloha_key_state_lora/<repo_id>/norm_stats.json。
4. norm stats smoke 命令使用和正式训练一致的 train_config_name 与 data.repo_id override。
5. 每个任务完成最小训练 smoke，5 step 训练。验证完整训练流程，特别是保存ckpt和训练配置，快照。
6. 训练 smoke 使用单独的 smoke exp_name，不污染正式训练 checkpoint 路径。
7. checkpoint metadata 保存 resolved train config、训练 command.txt 和 LeRobot repo 的 meta/rmbench/ 四文件快照。
```

提交要求：

```text
不提交 norm_stats 大文件或 smoke checkpoint。
提交 pi05 key-state runtime 支持、训练配置、批量训练入口。
```

### 7. 正式训练

第 6 阶段实现提交并通过 norm stats smoke 和训练 smoke 后，在 clean commit 上正式重算每个任务的 norm stats，然后使用 `experiments/pi0_key_state_baseline/` 下的入口启动四个任务的 pi0 key-state LoRA 训练。正式 norm stats 不带 `--max-frames`，直接覆盖第 6 阶段写入的 smoke `norm_stats.json`。wandb 使用：

```text
project: RMBench
group: pi0_key_state_baseline
job_type: train
```

验收标准：

```text
1. 每个任务训练完成到预期 step。
2. 每个任务的正式 norm_stats 已不带 `--max-frames` 重新计算，并覆盖 smoke norm_stats。
3. 每个 checkpoint 目录包含训练命令、commit、resolved config 或等价 metadata。
4. 每个任务都有 wandb id，wandb project/group/name 符合规范。
5. checkpoint metadata、wandb config/artifact 和 checkpoint 中保存的 meta/rmbench/ 四文件快照能互相追溯。
```

提交要求：

```text
不提交 checkpoint 或 wandb 本地目录。
正式训练启动后，更新并提交 `experiments/pi0_key_state_baseline/README.md`，写清楚每个任务的 checkpoint path、wandb id 或运行状态。
正式训练完成后必须先按验收标准自查，再次更新并提交 README，补充完成状态、最终 checkpoint、wandb id 和必要观察。
```

### 8. 正式评测与 README 更新 （未审核）

第 7 阶段训练完成并提交 `experiments/pi0_key_state_baseline/README.md` 完成记录后，先用同一 eval 入口和同一 checkpoint 跑小规模 eval smoke，例如 3 rollouts、1 个 overlay 视频，输出到 ignored smoke 目录。eval smoke 通过并确认正式运行相关 worktree 干净后，再在 clean commit 上对四个任务分别进行正式 eval，得到 pi0 + key-state SR。评测默认打开 key-state overlay 视频，结果按实验批次组织：

```text
eval_result/pi0_key_state_baseline/<run_id>/
```

验收标准：

```text
1. 每个任务完成 100 rollouts，前 5 个 rollout 录制 overlay 视频。
2. eval_result 目录包含 _result.txt、eval_log.txt、stdout.log、config.yaml、command.txt、key_state_config.yaml。
3. command.txt 记录 git commit、cwd、白名单 env 和启动命令。
4. overlay 显示当前任务的 phase/attribute label，不再写死 put_back_block 的 phase/mat。
5. `experiments/pi0_key_state_baseline/README.md` 记录每个任务的 checkpoint、eval result、wandb id、SR 和必要观察。
6. `experiments/pi0_key_state_baseline/README.md` 的主结果表只记录正式完成的训练和评测；smoke、失败启动、半成品不进入主结果表。
```

提交要求：

```text
不提交 eval_result 大文件或视频。
正式评测启动后，更新并提交 `experiments/pi0_key_state_baseline/README.md`，写清楚 checkpoint、eval result 目标路径和运行状态。
正式评测完成后必须先按验收标准自查，再次更新并提交 README，补充 SR、视频/结果路径、wandb id 和结论摘要。
```
