# Cover Blocks Key-State Design

## 背景

`cover_blocks` 是当前 pi0 key-state baseline 中差距最大的任务：

```text
pi0_lora_baseline:                 1/100 = 1%
pi0_lora_key_state 20000 step:     0/100 = 0%
pi0_lora_key_state 30000 step:     0/100 = 0%
Paper Mem-0:                       68%
Paper Pi0.5:                       0%
```

现有 key-state 设计把任务表示为：

```text
phase:
  cover_left_position
  cover_middle_position
  cover_right_position
  uncover_red_block
  uncover_green_block
  uncover_blue_block

attributes:
  red_pos
  green_pos
  blue_pos
```

one-hot 版本使用 18 个 key-state 维度，label-id 消融虽然把数值维度压到了 4 个标量，但语义仍然是 `6 phase + 3 color-position memory`。如果模型难以学到这个语义分解，单纯把 one-hot 换成 label-id 不一定能解决 `cover_blocks` 的失败。

新的假设是：执行侧不需要知道“当前是 uncover red 还是 uncover green”，只需要知道现在要做什么动作、操作哪个位置。颜色顺序只应该服务于状态转移，而不是直接作为执行状态的一部分。

## 任务事实

`envs/cover_blocks.py` 中的任务流程是固定的：

```text
1. 按位置顺序盖住 left, middle, right 三个 block。
2. 按颜色顺序 red, green, blue 打开对应位置的 cover。
```

环境成功判定依赖 cover mask 的严格转移：

```text
000 -> 100 -> 110 -> 111 -> open(red_pos) -> open(green_pos) -> open(blue_pos)
```

其中 `red_pos / green_pos / blue_pos` 由每个 episode 的随机颜色排列决定。`scene_info.json` 已经记录：

```text
task_facts.rgb_positions:
  red: left | middle | right
  green: left | middle | right
  blue: left | middle | right

task_facts.open_order_positions:
  [red_pos, green_pos, blue_pos]
```

## 设计目标

这个任务需要把两个问题分开：

```text
执行状态:
  当前机器人应该执行哪类动作，以及操作哪个空间位置。

状态转移记忆:
  当 block 被 cover 遮住以后，系统仍然要知道 red/green/blue 对应的打开位置。
```

执行侧可以压缩成两个标量：

```text
operation_id:
  cover | uncover

slot_id:
  left | middle | right
```

但是如果正式 eval 不允许读取 privileged scene_info，只保留这两个标量是不完整的。原因是：三个 block 全部被盖住以后，视觉上已经看不到 green / blue 的位置；如果状态里没有任何记忆，模型在打开 red 以后无法可靠知道下一个应该打开哪个 cover。

因此正式方案需要保留一个紧凑的状态转移记忆。这个记忆不直接表示执行动作，而是用于生成后续的 `operation_id / slot_id`。在当前 converter 体系里，这类记忆仍然应该使用现有 `attributes` 表示，不需要再引入一套新的 `memory_fields`。

## 推荐方案：Exec2 + Attributes

推荐先做一个新的 `label_id` 设计：

```text
dim 14: operation_id
  labels: [cover, uncover]

dim 15: slot_id
  labels: [left, middle, right]

dim 16: open_slot_0 attribute
  labels: [unknown, left, middle, right]
  meaning: red block 对应的 cover 位置

dim 17: open_slot_1 attribute
  labels: [unknown, left, middle, right]
  meaning: green block 对应的 cover 位置

dim 18: open_slot_2 attribute
  labels: [unknown, left, middle, right]
  meaning: blue block 对应的 cover 位置

dim 19:32
  padding zero
```

这个设计仍然有 5 个标量，比现有 label-id 的 4 个标量多一个，但它把语义改成了更贴近执行的形式：

```text
现有 label-id:
  phase_id + red_pos + green_pos + blue_pos

新设计:
  operation_id + slot_id + open_order_slots
```

关键差异不是数值维度从 4 到 5，而是把 6 类 phase 拆成了 2 类动作和 3 个空间位置。模型执行时只需要对齐当前操作目标，不需要同时学习 `uncover_red_block` 这种颜色语义和具体空间动作之间的映射。

## 执行标签生成

从现有 6 个 high-level phase 到新执行状态的映射如下：

| Existing phase | operation_id | slot_id |
| --- | --- | --- |
| `cover_left_position` | `cover` | `left` |
| `cover_middle_position` | `cover` | `middle` |
| `cover_right_position` | `cover` | `right` |
| `uncover_red_block` | `uncover` | `task_facts.rgb_positions.red` |
| `uncover_green_block` | `uncover` | `task_facts.rgb_positions.green` |
| `uncover_blue_block` | `uncover` | `task_facts.rgb_positions.blue` |

`operation_id / slot_id` 是 per-step 执行状态，时间边界仍然沿用现有 micro-stage / phase range。

## 记忆标签生成

`open_slot_0 / open_slot_1 / open_slot_2` 是 per-episode 的颜色顺序记忆。它们的 target 来自：

```text
open_slot_0 = task_facts.open_order_positions.0
open_slot_1 = task_facts.open_order_positions.1
open_slot_2 = task_facts.open_order_positions.2
```

推荐沿用当前 `cover_blocks` 的 acquisition window：

```text
update_window:
  [episode_start, micro_stages.cover_left_position_place.start_frame)
```

窗口含义：

```text
窗口前和窗口内:
  state input 为 unknown。
  action target 为对应 slot。

窗口后:
  state input 和 action target 都锁存为对应 slot。
```

这个窗口结束点不能延到第一个 cover 已经放下之后。否则 left block 被遮住后，模型仍然需要从不可见颜色里预测完整 open order，会混入不必要的集合推理。

## 为什么不直接用纯二维状态

纯二维状态如下：

```text
operation_id
slot_id
```

它可以作为诊断实验，但不应该直接作为正式结果，除非明确说明状态转移使用了 privileged 信息。

可行的诊断版本：

```text
cover_blocks_exec2_oracle_transition:
  policy 只接收 operation_id / slot_id。
  eval runtime 用 scene_info 或环境内部真值决定 red/green/blue 的打开位置。
```

这个版本可以回答一个问题：如果状态转移完全正确，二维执行状态是否足以让机器人完成动作。但它不公平，因为正式策略部署时不能读取 episode 的 `scene_info` 真值。

正式可比版本应该使用：

```text
cover_blocks_exec2_attr3_no_phase:
  policy 接收 operation_id / slot_id。
  policy 同时预测并锁存 open_slot_0 / open_slot_1 / open_slot_2。
  eval runtime 只使用模型已经锁存的 open_order attributes 生成后续执行状态。
```

## Converter Schema

当前 converter 支持一个 `phase` block 和若干 `attributes` block。新设计不应该替换 `attributes`，只需要新增一个 `execution` block：

```text
execution:
  per-step 标签，按 phase range 直接赋值。
  这里包括 operation_id 和 slot_id。

attributes:
  带 update_window 的属性记忆。
  这里包括 open_slot_0 / open_slot_1 / open_slot_2。
```

`phase` 可以保留为可选字段，用于做消融：

```text
no_phase:
  execution + attributes。
  检查二维执行状态加紧凑记忆是否足够。

with_phase:
  phase + execution + attributes。
  检查显式 progress 信号是否仍然有帮助。
```

这个改动比引入 `execution_fields / memory_fields` 更小，也和现有 converter 的概念一致：`phase` 表示进度，`attributes` 表示需要推断和锁存的 episode-level 信息，`execution` 表示当前要执行的动作目标。

## Runtime 要求

runtime 需要同步支持 `label_id` 字段：

```text
1. label_id action 不能使用 one-hot argmax 解码。
2. 应按 round + clip 或最近 label 解码为离散 id。
3. attributes 中的 unknown 只允许向非 unknown 转移一次，之后锁存。
4. execution 字段可以按模型输出更新，也可以由一个通用 transition controller 根据 progress 和 attributes 生成。
```

如果先做最小实现，建议仍然让 policy 直接预测 `operation_id / slot_id`，runtime 对每个字段做离散化和锁存；不要在 runtime 中写死 cover_blocks 的红绿蓝逻辑。需要任务依赖规则时，应写进 key-state config，而不是写死在 pi0/pi05 推理代码里。

## Config 草案

下面是目标语义，不要求当前 converter 立即兼容该字段名：

```yaml
dataset:
  task: cover_blocks
  source_dir: data/cover_blocks/demo_clean_state
  repo_id: cover_blocks_demo_clean_state_key_state_exec2_attr3_no_phase
  episodes: 50

state_layout:
  state_dim: 32
  robot_dim: 14

execution:
  - name: operation
    dim: [14, 15]
    encoding: label_id
    labels: [cover, uncover]
    ranges:
      - label: cover
        window: [episode_start, micro_stages.cover_right_position_release.end_frame]
      - label: uncover
        window: [micro_stages.cover_right_position_release.end_frame, episode_end]

  - name: slot
    dim: [15, 16]
    encoding: label_id
    labels: [left, middle, right]
    ranges:
      - label: left
        window: [episode_start, micro_stages.cover_left_position_release.end_frame]
      - label: middle
        window: [micro_stages.cover_left_position_release.end_frame, micro_stages.cover_middle_position_release.end_frame]
      - label: right
        window: [micro_stages.cover_middle_position_release.end_frame, micro_stages.cover_right_position_release.end_frame]
      - label: task_facts.open_order_positions.0
        window: [micro_stages.cover_right_position_release.end_frame, micro_stages.uncover_red_block_release.end_frame]
      - label: task_facts.open_order_positions.1
        window: [micro_stages.uncover_red_block_release.end_frame, micro_stages.uncover_green_block_release.end_frame]
      - label: task_facts.open_order_positions.2
        window: [micro_stages.uncover_green_block_release.end_frame, episode_end]

attributes:
  - name: open_slot_0
    dim: [16, 17]
    encoding: label_id
    labels: [unknown, left, middle, right]
    transitions:
      - from_value: unknown
        to_value: task_facts.open_order_positions.0
        update_window: [episode_start, micro_stages.cover_left_position_place.start_frame]

  - name: open_slot_1
    dim: [17, 18]
    encoding: label_id
    labels: [unknown, left, middle, right]
    transitions:
      - from_value: unknown
        to_value: task_facts.open_order_positions.1
        update_window: [episode_start, micro_stages.cover_left_position_place.start_frame]

  - name: open_slot_2
    dim: [18, 19]
    encoding: label_id
    labels: [unknown, left, middle, right]
    transitions:
      - from_value: unknown
        to_value: task_facts.open_order_positions.2
        update_window: [episode_start, micro_stages.cover_left_position_place.start_frame]
```

## 实验对比

建议把这批实验和当前 `pi0_key_state_encoding_ablation` 做同表对比：

```text
1. current_one_hot:
   6 phase one-hot + red/green/blue one-hot。

2. current_label_id:
   6 phase label-id + red/green/blue label-id。

3. exec2_attr3_no_phase:
   operation/slot label-id + open_order attributes label-id。

4. phase_exec2_attr3:
   phase label-id + operation/slot label-id + open_order attributes label-id。

5. exec2_oracle_transition:
   只作为诊断上界，不进入正式主表，除非表格明确标注 oracle。
```

核心判断：

```text
如果 current_label_id 仍然接近 0，而 exec2_attr3_no_phase 明显提升，
说明问题主要来自 key-state 语义设计，而不是 one-hot 维度本身。

如果 phase_exec2_attr3 明显优于 exec2_attr3_no_phase，
说明 cover_blocks 仍然需要显式 progress 信号；反之可以去掉 phase，保持状态更紧凑。

如果 exec2_oracle_transition 明显提升，但 exec2_attr3_no_phase 和 phase_exec2_attr3 仍然失败，
说明主要问题在颜色顺序记忆的获取和锁存，而不是执行状态本身。

如果 exec2_oracle_transition 也失败，
说明失败更可能来自底层动作学习、cover 抓取/放置精度、语言截断或训练数据规模。
```

## 下一步

1. 扩展 converter，使其在现有 `phase / attributes` 之外支持 `execution`。
2. 扩展 pi05 runtime 的 key-state schema 解析和 overlay，支持 `label_id` 字段。
3. 写 `cover_blocks_exec2_attr3_no_phase.yaml` 和可选的 `cover_blocks_phase_exec2_attr3.yaml`，先做 1 episode conversion smoke。
4. smoke 验收 state/action：
   - `operation_id` 只出现 `0/1`。
   - `slot_id` 只出现 `0/1/2`，且 uncover 阶段等于 `open_order_positions`。
   - `open_slot_0/1/2` 在 acquisition window 后锁存。
   - dim 19:32 全部为 0。
5. 正式转换 50 episodes，计算 norm stats，训练 pi0 lora。
6. 评测 100 rollout，前 5 个录制 overlay 视频，和 current label-id 消融一起比较。
