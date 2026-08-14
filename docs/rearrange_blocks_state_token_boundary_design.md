# Rearrange Blocks：State Token 与 Action Boundary 消融设计

## 1. 文档状态

- 状态：首版已实现；2×2 smoke 已通过，formal training 待运行。
- 任务范围：第一阶段只研究 `rearrange_blocks`。
- 目标模型：Pi0.5 full finetune。
- 核心实验：并行/串行 state 输出 × Soft/Hard action boundary，共 2×2 四组。
- 本文同时记录已落地的语义、数据、模型、rollout 与实验实现，作为后续实现漂移的评审基线。

## 2. 背景与问题

当前方案把 key state 作为连续维度同时注入 Pi0.5 的 robot state observation 和
action target。新方案希望把任务状态从机器人本体状态和连续动作中分离出来，改为模型
显式输入和输出的离散 state token。

`rearrange_blocks` 当前成功率较低，而且已经观察到一种常见失败：机器人完成了类似
按按钮的运动，但实际下压不到位，环境中的 `press_cnt` 仍为 0。这个现象同时暴露了两个
相互关联、但必须分开控制的设计问题：

1. state 是和 action 并行预测，还是先预测 state，再用 state 条件化 action；
2. action chunk 是否允许跨越需要重新观察确认的边界。

此前对 phase 边界做标签抖动式数据增强后效果变差。本文不再把边界不确定性直接写入
监督标签，而是先固定唯一的时刻定义，并通过正交消融判断问题究竟来自状态因果结构、
chunk 跨界，还是按钮本身不可观测/接触时间不足。

现有实现和数据定义见：

- [`key_state_rearrange_swap_battery_cover_design.md`](key_state_rearrange_swap_battery_cover_design.md)
- [`rearrange_blocks.yaml`](../converter_configs/key_state_baseline/rearrange_blocks.yaml)
- [`rearrange_blocks.py`](../envs/rearrange_blocks.py)
- [`pi05_full_key_state/README.md`](../experiments/pi05_full_key_state/README.md)

## 3. 任务状态与环境事实

### 3.1 现有高层 phase

保持现有的三个高层 phase，不因按钮 guard 而整体重写任务结构：

| ID | Phase | 现有标注区间 |
| --- | --- | --- |
| P0 | `move_middle_block_to_empty_mat` | episode start → `block1_place.end_frame` |
| P1 | `press_button_after_first_move` | `block1_place.end_frame` → `press_return.end_frame` |
| P2 | `move_original_mat_block_to_middle` | `press_return.end_frame` → episode end |

已有属性 `empty_mat_side ∈ {unknown, left, right}` 继续保留，并在确定后锁存。

### 3.2 按钮动作的真实结构

环境目前把以下动作都包在同一个 `press_button` micro-stage 中：

1. 靠近按钮；
2. 下压；
3. 调用成功检查；
4. 抬起释放；
5. 重置按钮。

因此，现有 `press_button.end_frame` 已经晚于“按钮被压下”这个关键时刻。若直接在该
边界截断 action chunk，截断后的最后动作很可能是抬起/释放，而不是保持下压，无法针对
`press_cnt == 0` 的失败。

本文因此区分两类边界：

- **phase boundary**：任务语义从 P0/P1/P2 中的一个切换到另一个；
- **guard boundary**：继续后续动作前，需要重新观察某个执行条件是否成立。guard 不一定
  改变高层 phase。

首轮 Hard 实验只启用一个 guard：`button_press_confirmed`。不同时把
`block1_place.end_frame` 也设为 Hard，以免把“第一块放置确认”和“按钮下压确认”两个
因素混在一个实验变量中。

## 4. 统一的 state 时刻定义

对第 `q` 次 policy query，令当前物理观测时刻为 `t_q`：

```text
m_q^-  : query 进入时携带的上一次状态
o_q    : t_q 时刻的当前 observation
m_q^+  : 读完 o_q 后、生成新 action chunk 前的状态
A_q    : 从 t_q 开始执行的 action chunk
```

统一更新关系为：

```text
m_q^+ = Update(o_q, m_q^-)
A_q   ~ Policy(o_q, m_q^+)
```

数据集中 state output target 永远定义为 `m_q^+`，即：

> 当前 observation 所处的执行上下文，也是下一段动作开始时应该采用的 state。

它不表示整个 action chunk 都必须留在这个 phase，也不表示 chunk 执行完成后的未来
state。于是当真实 phase 在物理时刻 `T` 变化时：

- `t_q < T`：state target 仍为旧 phase；
- `t_q >= T`：只有当前 observation 已经来自新状态，state target 才切换；
- Soft chunk 可以从 `t_q < T` 开始并跨过 `T`；这不要求提前把 state target 标成新 phase。

这消除了“为了进入新 phase 必须先执行旧时刻输出的动作，但 state 又只能在看到新画面后
变化”的表面矛盾：**state 描述 query 起点，action 描述可跨状态的未来轨迹。**

## 5. State token schema

### 5.1 推荐的因子化 structured token

不把所有状态组合压成一个大枚举，也不把 phase、side、button 三个 embedding 相加成一个
token。三个字段各占一个独立 token slot：

```text
phase:               P0 | P1 | P2
empty_mat_side:      unknown | left | right
button_press_status: NA | unconfirmed | confirmed
```

其中：

- `phase`：当前 query 起点的高层执行上下文；
- `empty_mat_side`：任务属性，确定后锁存；
- `button_press_status`：P1 内的 guard 子状态；P0/P2 为 `NA`，P1 下压成功前为
  `UNCONFIRMED`，首次确认成功后为 `CONFIRMED` 并锁存到离开 P1。

增加 `button_press_status` 的原因是：按钮下压确认发生在 P1 内部，仅用三个高层 phase
无法表达“尚未确认，应继续压”与“已经确认，可以释放”。它是 guard state，不需要人为
增加第四个高层 phase。

首版不要求把这些状态注册成 SentencePiece/PaliGemma 的全局 vocabulary token。它们使用
字段局部的小词表和 learnable embedding table，避免扩大全局 tokenizer 与 checkpoint
embedding matrix。离散 category ID 加上对应 learnable embedding，仍然构成真正的
structured token。

### 5.2 Learnable 参数与 token 构造

对字段 `j ∈ {phase, side, button}`，定义：

```text
Q_j                       learnable field-query embedding
F_j                       learnable field/type embedding
E_j ∈ R^(K_j × d)         learnable value embedding table
S_prev, S_current         learnable state-segment embedding
```

首版令 `d` 等于 PaliGemma/VLM expert hidden width，使 state token 直接存在于 VLM prefix
空间。当前 Pi0.5 的 action expert 通过联合 Transformer attention 读取这些 prefix KV，
不另外把 state 扩展成连续 action 维度。

输入的 previous-state token 为：

```text
x_j^- = F_j + E_j[value_j^-] + S_prev
```

预测后写入 action context 的 current-state token 为：

```text
x_j^+ = F_j + E_j[value_j^+] + S_current
```

因此一次 query 中始终有三个独立 state tokens：

```text
[x_phase, x_side, x_button]
```

“embedding 相加”只发生在同一个字段 token 内，类似普通 Transformer 的 token embedding
+ position/segment embedding；禁止把三个语义字段压成 `x_phase + x_side + x_button`。

### 5.3 三字段并行预测与 input/output weight tying

三个字段没有自然的全序关系：`empty_mat_side` 基本静态，`button_press_status` 受到 phase
约束，但不需要让 `phase → side → button` 形成三步 AR。首版使用一次 field-query block
并行预测三个字段：

```text
h_phase, h_side, h_button =
    StateTransformer(context, [Q_phase, Q_side, Q_button])
```

每个字段的输出分类权重与其 value embedding table 绑定：

```text
z_phase  = Norm(h_phase)  @ E_phase.T  + b_phase
z_side   = Norm(h_side)   @ E_side.T   + b_side
z_button = Norm(h_button) @ E_button.T + b_button
```

`z_j` 是 raw classification logits。state loss 直接使用稳定的 softmax cross-entropy：

```text
L_state = λ_phase  CE(z_phase,  y_phase)
        + λ_side   CE(z_side,   y_side)
        + λ_button CE(z_button, y_button)
```

实现中把 raw logits 传给 cross-entropy，不预先手写 softmax。由于 `E_j` 同时出现在
`z_j = h_j E_j^T` 和 state token lookup 中，state CE 与 action loss 都能训练同一份 value
embedding。这与语言模型共享 input embedding/LM head 权重的做法一致。

不采用独立的 `Linear(d, K_j)` 分类权重 `W_j` 再配另一套 action embedding `E_j`；否则
“识别出 P1”和“用 P1 条件化 action”会落在两套没有绑定的表示空间。

### 5.4 输入 token 与输出 token 的数据时刻

- 输入 state token 对应 `m_q^-`，即上一次 query 输出并携带到本次的状态。对
  `rearrange_blocks`，episode 首次 query 固定初始化为
  `m_0^- = (P0, EMPTY_UNKNOWN, BUTTON_NA)`，与普通 previous state 一样使用 `S_prev`。
- 输出 state token 对应 `m_q^+`，由当前 observation 和输入 state 共同更新。
- offline 数据必须按实际 rollout query stride 构造 `m_q^-`，而不是把同一时刻的真值
  state 同时作为输入和 target；否则模型只需复制 token，无法学习状态更新。
- rollout 时只携带模型自身上一次的预测，禁止使用环境真值回填。环境真值只允许出现在
  oracle 诊断中。

若 rollout 固定每 `S` 个 control steps 查询一次，固定-lag 的离线 teacher forcing 为：

```text
state_input[t]  = state_target[t - S],                 t - S >= episode_start
                = (P0, EMPTY_UNKNOWN, BUTTON_NA),      otherwise
state_target[t] = state_at_current_observation[t]
```

固定 `S` 会让模型只看到单一 memory age。随机-lag 训练允许在 data config 中设置闭区间
`key_state_previous_lag_range = (S_min, S_max)`；每次读取样本都重新采样：

```text
L_t ~ UniformInteger(S_min, ..., S_max)
state_input[t]  = state_target[t - L_t],               t - L_t >= episode_start
                = episode_initial_state,               otherwise
```

`rearrange_blocks` 的首个随机-lag 实验使用 `[15, 50]`。随机化仅作用于训练输入的
previous-state token，不改变 current-state target、action supervision、action horizon、共享
LeRobot dataset 或 norm stats；推理仍携带上一次真实 query 的模型输出，而不是人为采样 lag。
实现时必须按 episode 边界初始化，禁止从上一条 episode 泄漏 previous state。这里不增加
独立的 `S_init`：`S_prev/S_current` 只表达 token 在更新前还是更新后，初始化值由任务状态
本身表达。未来若某任务没有已知初态，应使用显式 input-valid mask 或 input-only
`NO_PREVIOUS_STATE` value，而不是再增加一个与 segment 语义重叠的 embedding。

## 6. 结构轴：Parallel 与 Serial

### 6.1 Parallel：状态和动作并行输出

分解近似为：

```text
p(m_q^+, A_q | o_q, m_q^-)
```

共享 observation/context 表征后，field-query/tied-logit 模块和 action expert 分别输出。
Parallel 不把选中的 current-state value embeddings 作为新 block 写回上下文；Serial 则增加
这个 block。两者保持相同的原生 block-causal attention 规则。

优点是单次前向、延迟低，错误 state 不会直接硬性污染 action；缺点是 state 可能退化成
辅助任务，模型仍然绕过 token，直接从视觉预测动作。

### 6.2 Serial：先输出状态，再条件化动作

分解为：

```text
p(m_q^+ | o_q, m_q^-)
p(A_q | o_q, m_q^-, m_q^+)
```

先并行预测三个 state 字段，再把选中的三个 value embeddings 写入 action context。这里的
“Serial”只表示 `current state → action` 的显式因果边，不表示三个 state 字段之间采用
三步 autoregressive decoding。

### 6.3 Serial 的 Transformer/KV 实现

将一次 query 划分为四个 attention block：

```text
A. context:
   images + instruction + previous-state tokens

B. state queries:
   Q_phase + Q_side + Q_button

C. selected current-state tokens:
   x_phase^+ + x_side^+ + x_button^+

D. action suffix:
   noisy continuous action tokens + flow timestep
```

不增加定制 attention hole，直接复用 Pi0.5 原有的 block-causal 规则：每个 block 可以
attention 所有之前的 block，也可以 attention 自己 block 内的其他 token，但看不到之后的
block：

```text
A reads: A
B reads: A + B
C reads: A + B + C
D reads: A + B + C + D
```

三个 field queries 位于同一个 B block，因此并行计算并能相互交换 hidden information；它们
不以离散预测类别作为彼此输入，不构成字段级 AR。Parallel 直接省略 C，所以其 D 读取
`A+B+D`；Serial 插入 C，所以其 D 读取 `A+B+C+D`。

这意味着 action 在两组中都能读取相同的 B query hidden states，因此本实验不是“只允许
离散 state 通过的严格信息瓶颈”。但这一共享路径在 Parallel/Serial 中完全一致，结构轴的
唯一增量仍是 Serial 是否把选中的离散 current-state embeddings C 写回 action context。
这样能回答离散 state conditioning 是否有额外价值，同时保持对现有 attention mask 的最小
改动。若后续确实需要研究 strict bottleneck，再单独增加屏蔽 B 的实验，不混入首轮 2×2。

当前 Pi0.5 rollout 已经先计算 prefix KV cache，再让 action suffix 做多步 flow-matching
denoising。Serial 在此基础上执行：

1. 计算 A+B，一次性得到三个字段 logits 和 prefix KV；
2. 对三个字段并行做约束选择，得到 category IDs；
3. lookup 同一组 `E_j`，把 C 追加到 VLM prefix KV；
4. Parallel 的 action denoising 读取 A+B cache，Serial 读取 A+B+C cache；
5. 把三个 category IDs 保存为下一次 query 的 `m^-`。

这不是三步 AR decode，也不需要把图像重新编码三次。Parallel 与 Serial 都计算 B 以输出
state；Serial 相对 Parallel 只增加离散选择同步、C 的三个 condition tokens 及对应 KV。

### 6.4 Softmax、hard token 与梯度路径

训练 state classifier 时直接对 raw logits 计算 CE。正式 rollout 时先应用字段合法类别/
状态转移 mask，再选择：

```text
value_j^+ = argmax(masked_z_j)
x_j^+     = F_j + E_j[value_j^+] + S_current
```

`argmax(z) == argmax(softmax(z))`，所以类别选择不需要显式 softmax。softmax 概率只用于
离线评测和校准诊断，不参与 rollout 决策。phase、side、button 三个字段使用完全相同的
更新逻辑：先按 previous state 应用合法转移 mask，再对 masked logits 做 argmax；button
不设置额外置信度阈值。

仿真中的按钮关节/接触阈值只负责产生 `BUTTON_CONFIRMED` 监督标签并定位物理边界 `T`，
不进入模型 rollout 的状态转移逻辑。

首版合法转移固定为：

```text
phase:  P0 → {P0, P1}; P1 → {P1, P2}; P2 → {P2}
side:   unknown → {unknown, left, right}; left → {left}; right → {right}
button: phase != P1 时强制 NA；
        P1 内 unconfirmed → {unconfirmed, confirmed}; confirmed → {confirmed}
```

评测同时记录 mask 前 raw prediction 和 mask 后 executed state，避免规则掩盖模型本身的分类
错误。

主 2×2 实验使用 hard category token，不把概率加权的 soft embedding 输入 action，也不使用
straight-through/Gumbel estimator，避免 action 通过连续概率分布携带类别之外的信息。

各参数的主梯度来源为：

| 参数 | State CE | Action flow loss |
| --- | --- | --- |
| `Q_j` field query | 是 | 是，action 通过 B 的共享 attention 路径读取 |
| `E_j` value embedding | 是，作为 tied output weight | 是，作为 action condition |
| `F_j, S_*` token metadata | 是，经 previous state 输入 | 是，经 current state condition |
| shared observation context A | 是 | 是，action 直接读取 A |
| state query hidden B | 是 | 是，Parallel/Serial 共有路径 |
| hard category selection | state CE 训练其 logits | 训练时 teacher forcing；推理时 argmax |
| action expert | 否 | 是 |

Serial 训练采用 Transformer 标准 teacher forcing，在一次 block-causal forward 中排列：

```text
A(observation, previous state) → B(state queries) → C(GT state embeddings) → D(actions)
```

B 的 hidden 产生 state logits 和 `L_state`；D 能看到 C，因此 action supervision 明确以当前
state 为条件。推理时先运行 A+B，做合法转移 mask 与 argmax，再把预测 state embeddings 作为
C 追加到 KV cache 后生成 D。三个语义字段仍是同一个 block，不做字段级 AR。

首版主 2×2 不做 scheduled predicted conditioning。双前向 scheduled sampling 在 batch size 32
下会同时保留两套 Pi0.5 反向图，实测超过 80 GB；teacher forcing 则与标准 next-token 训练一致，
并保持一次前向和最小结构改动。代价是 Serial 存在 exposure bias，必须通过 predicted/oracle
双评测单独诊断。action loss 会更新 C 中被选 GT 类别的 embeddings；完整词表和分类边界仍由
`L_state` 直接训练。这里保留的是可解释的离散 conditioning 路径，而不是严格阻断所有连续
state-query information 的 bottleneck。

### 6.5 Serial 诊断

必须同时报告两种 Serial rollout：

- `predicted-state`：正式结果，使用模型预测 token；
- `oracle-state`：只作诊断，使用环境真值 token，给出 state 分类错误之外的 action 上限。

若 oracle-state 明显优于 predicted-state，瓶颈是 state 感知；若二者都差，瓶颈更可能在
action 或 token 条件化机制。

当前实现通过评测配置 `state_token_rollout_mode: predicted | oracle` 切换。`oracle` 仅允许
Serial checkpoint，并遵循以下诊断契约：

1. A block 中的 previous state 来自上一次 query 的 oracle state；
2. B block 仍正常输出模型预测和 logits；
3. C block 使用当前环境 oracle state，因此 action 完全以 GT current state 为条件；
4. 每次 query 同时记录 `predicted_ids` 和实际用于控制的 `executed_ids`，便于后续计算分类
   错误和 oracle gap。

`rearrange_blocks` 的在线 oracle 不读取未来 expert 轨迹。它只读取仿真物理状态，并保持 phase
单调：第一块已经正确放置且右夹爪释放后从 P0 进入 P1；一次有效按钮按压完成、释放且左臂回到初始位姿后从 P1
进入 P2。`empty_mat_side` 直接取 episode task fact；P1 内按钮未确认/已确认分别由环境现有的
`press_cnt/press_flag` 真值决定，P0/P2 使用 `NA`。这与离线 query-time state 的语义一致，
同时避免用未来 frame 或演示时间索引作弊。

Full key state 另通过 `key_state_rollout_mode: predicted | oracle` 做同语义诊断。Dense oracle
不修改 Transformer：在每个观测进入 policy 前，使用同一个 checkpoint schema 把环境 GT 字段编码为
32D state 的 one-hot memory tail；action 的前 14D 仍正常控制机器人，后 18D 中受 schema 管理的
memory 预测不再递推到后续观测。chunk 内部保存观测时也重新读取当前环境 GT，因而不会把某个
action step 的预测 memory 混入下一次 query。未被 schema 使用的 padding 维保持 0。默认值为
`predicted`，现有 full/state-token checkpoint 的行为保持不变。

### 6.6 与 Mealy/Moore 的关系

这个类比有帮助，但不完全等价。Parallel/Serial 更准确地说是联合预测与层级条件分解的
区别；Mealy/Moore 描述输出依赖输入还是只依赖内部状态。Hard boundary 则更接近 options
框架里的 termination/guard。本文因此不直接把四个实验命名为 Mealy/Moore。

### 6.7 现有 Pi0/Pi0.5 兼容性契约

state-token 必须作为 opt-in 扩展接入现有实现，不能把当前 Pi0/Pi0.5 主路径整体替换掉。
兼容性要求如下。

#### 6.7.1 Config 与参数树

在 `Pi0Config` 增加显式开关，默认值必须关闭：

```text
key_state_token_mode: disabled | parallel | serial = disabled
key_state_schema: optional structured config = None
```

- 所有现有 train config 不改配置即可继续实例化；
- `disabled` 时不创建 `Q_j/F_j/E_j/S_*` 等任何新参数，flatten 后的参数 key、shape 和 dtype
  必须与修改前完全一致；
- `disabled` 时 `embed_prefix`、`compute_loss` 和 `sample_actions` 走原有分支，固定输入/RNG 下
  输出保持数值一致；
- 首版 state-token 只支持 `pi05=True`；若 Pi0/Pi0-FAST/PyTorch 路径显式启用，必须 fail
  fast，而不是静默退化。关闭时这些模型完全不受影响；
- `pi05_state_sequence_in_suffix` 等现有选项保持原语义，不被 state-token 开关覆盖。

#### 6.7.2 Observation 与数据接口

不改变现有 robot `state`、连续 `actions` 或 `tokenized_prompt` 的含义和维度。key state 通过
`Observation` 末尾新增的 optional sidecar 字段传递，例如：

```text
key_state_input_ids:   int32[B, num_fields] | None
key_state_target_ids:  int32[B, num_fields] | None
key_state_target_mask: bool[B, num_fields]  | None
```

- 旧数据 transform 不产生这些字段，`Observation.from_dict` 默认读为 `None`；
- `preprocess_observation` 必须原样透传 optional sidecar；
- `disabled` 的 `inputs_spec` 不增加新的 array leaf；
- `Observation.to_dict` 不输出值为 `None` 的新字段，保持旧 policy/client 字典接口；
- state target 不重新塞回 action 维度，因此旧 action norm stats、action horizon 和 action
  decoder 均不改变。

#### 6.7.3 Loss API

现有 `BaseModel.compute_loss` 返回 `[*batch, action_horizon]`，首版不修改该公共签名：

```text
disabled:
    total_loss[..., k] = original_action_loss[..., k]

enabled:
    total_loss[..., k] = action_loss[..., k] + λ_state * state_loss
```

外层仍按原方式求均值。state/action 分项指标通过 opt-in aux/metrics helper 记录，不迫使旧
trainer 接受新的返回类型。`disabled` 分支不得执行 state-token loss 代码。

#### 6.7.4 Checkpoint 加载

- 旧 config + 旧 checkpoint：严格恢复原参数树，不允许出现新增 missing keys；
- 新 state-token config 从 `pi05_base` 初始化：只允许显式白名单 namespace（例如
  `.*key_state_token.*`）缺失并随机初始化，不能把 `missing_regex` 放宽成 `.*`；
- 新 state-token checkpoint resume：要求 state schema、mode 和全部新增参数严格匹配；
- checkpoint metadata 必须记录 mode、字段顺序、各字段类别表与 Serial conditioning 方式
  （本批次为 teacher forcing）；不匹配时 fail fast，禁止静默重排 category IDs；
- eval/deploy 必须从被评测 run 的 `metadata/train_config.yaml` 恢复 `key_state_token_mode`
  和字段类别数；共享 train config 中用于 CLI 的默认 mode 不是 Serial checkpoint 的推理依据；
- loader 必须打印并保存所有被初始化的 missing parameter names，方便审计。

#### 6.7.5 推理与 Policy API

保留现有：

```text
sample_actions(...) -> actions
```

另加 opt-in 方法：

```text
sample_actions_with_key_state(...)
    -> actions, current_state_ids, state_logits
```

- 旧 `Policy` 继续调用 `sample_actions`，输出字典仍至少保持原来的 `state/actions`；
- state-token rollout wrapper 只在新 mode 下调用 aux 方法，并额外输出 `key_state` 诊断；
- previous-state memory 由 rollout wrapper 持有，在 episode reset 时恢复固定初态，不把可变
  episode 状态存进 model parameters；
- `sample_actions` 在 state-token 模型上仍可作为兼容 wrapper，仅返回 actions；正式
  state-token rollout 必须使用 aux 方法以取得下一次 query 的 state IDs。

#### 6.7.6 必须通过的回归测试

1. 所有现有 Pi0/Pi0.5 config 仍能构建、训练一步和运行 `sample_actions`；
2. `disabled` 模式的 flattened parameter tree 与修改前完全相同；
3. 固定 checkpoint、observation、noise 和 RNG，`disabled` 前后 action 输出一致；
4. 旧 checkpoint 在旧 config 下无 missing/unexpected state-token params；
5. `pi05_base → state-token config` 只初始化白名单内的新参数；
6. state-token checkpoint 用错误字段顺序/类别数加载时明确失败；
7. 旧 data transform、policy server/client 和 output transform 忽略 optional key-state 字段后
   行为不变；
8. Parallel/Serial 两种新模式分别通过 JIT、单步 loss、checkpoint round-trip 和 rollout
   episode-reset 测试。

## 7. 数据轴：Soft 与 Hard action boundary

### 7.1 Soft：允许 chunk 跨 guard

保持原始示范 action chunk：

```text
A_soft[t, k] = a[t + k]
```

一个 chunk 可以包含靠近、下压、释放，甚至进入后续 phase 的动作。其优势是动作连贯，
而且失败后有自然重试行为：如果执行完 chunk 后视觉仍类似未完成状态，无记忆策略会再次
给出相应动作。

风险是 chunk 内后半段动作不会等待新的 observation。按钮没有压到位时，策略也可能继续
释放和返回，从而漏掉本应发生的确认。

### 7.2 Hard：到 guard 后保持，等待下一次 query

令 `T` 是成功示范中按钮首次满足物理按下阈值后的第一帧，`a[T-1]` 是达到按下位置的最后
一个动作。对于 `t < T` 且原 action horizon 跨过 `T` 的样本：

```text
A_hard[t, k] = a[t + k],  t + k < T
A_hard[t, k] = a[T - 1],  t + k >= T
```

即把：

```text
..., old-3, old-2, old-1, new-1, new-2, ...
```

改成：

```text
..., old-3, old-2, old-1, old-1, old-1, ...
```

这里的 `old-1` 必须是“保持按钮下压”的动作，不能是释放动作。chunk 执行完重新 query：

- 若当前 observation 支持 `BUTTON_CONFIRMED`，Serial 模型可条件化输出释放/返回动作；
- 若仍为 `BUTTON_UNCONFIRMED`，策略应再次下压或调整，而不是进入后续动作。

为了保持 2×2 可比，Soft/Hard 使用完全相同的 state target；二者只改变 action chunk target。

### 7.3 为什么首版不用变长 action horizon

变长 horizon 语义更自然，但会同时引入 stop prediction、padding/mask、动态执行长度和 JAX
静态 shape 等额外变量。首版采用固定 shape 的 repeat-last 监督。

若首轮结果支持 Hard，再升级为：

```text
fixed padded action tensor + action_valid_mask + stop_offset
```

训练时只在有效区间计算 action loss，rollout 按 `stop_offset` 提前 requery。

## 8. 2×2 主实验

| Cell | State/action 结构 | Chunk boundary | 关键问题 |
| --- | --- | --- | --- |
| P-S | Parallel | Soft | state 作为辅助输出、保持原始连贯 chunk 的基线 |
| P-H | Parallel | Hard button guard | 只靠当前视觉和硬停顿，是否已能改善按钮失败 |
| S-S | Serial | Soft | state 是否能改善动作，但仍允许 chunk 跨 guard |
| S-H | Serial | Hard button guard | 显式确认后再释放是否最好 |

核心假设：

- `P-H > P-S`：硬 guard 本身有价值；
- `S-S > P-S`：state 对 action 的显式因果条件有价值；
- `S-H > P-H` 且 oracle gap 小：学到的确认 token 真正参与了控制；
- `P-H ≈ S-H`：Hard 的收益主要来自 dwell/requery，未必需要 Serial token；
- Hard 组 `press_cnt==0` 降低但总成功率不升：按钮问题缓解了，但停顿或后续恢复引入了新失败。

## 9. 数据与标注要求

### 9.1 必需的新事件标注

现有数据只有整体 `press_button` micro-stage，不能从 HDF5 物理量可靠恢复按下与释放之间的 `T`。正式实验
前，示范生成至少要拆出：

```text
button_approach
button_press_down
button_release
button_return
```

并额外记录：

```text
guard_events.button_press_confirmed.frame
guard_events.button_press_confirmed.source
```

`frame` 应由仿真按钮关节/接触阈值在采集时产生，而不是依赖人工观看视频估计。

当前首轮实现没有使用粗粒度 `press_button.end_frame`。仓库已有的
`language_annotation.json` 恰好包含 11 个示范程序分段，其中第 3/4/5/6 个零基 segment
分别对应 approach / press-down / release / return；实现暂取 segment 4 的末帧作为 `T`，并把
来源记录为 `language_annotation.segment_4.end`。这比整体 micro-stage 更细，足以运行结构消融，
但仍不是按钮关节/接触阈值产生的物理真值。结果若支持 Hard，下一轮仍应补采物理事件后复验。

### 9.2 边界不确定性处理

主 2×2 实验不使用边界 jitter，避免重现此前“增强后监督互相冲突”的问题。后续若验证
鲁棒性，只允许：

- 对 `m_q^-` 做 lag、dropout 或相邻状态替换；
- 保持 `m_q^+` 和 robot action target 不变；
- 对视觉上不可判定的窄边界带 mask/downweight state classification loss。

禁止把同一 observation 随机标成 old/new phase，同时仍保留原 action target。这会制造
一个输入对应两个相反控制语义的监督冲突。

### 9.3 归一化与维度

- robot state/action 恢复为纯机器人连续量，不再承载 key-state 连续维度；
- state token 为离散分类目标，不进入 action norm stats；
- Soft/Hard 共用同一份按原始连续 chunk 计算的 norm stats；Hard 只改变在线 action supervision，
  不让归一化尺度成为额外实验变量；
- converter 输出 state 输入、state target、guard frame、action mask/hold 比例等可审计字段。

## 10. 训练与 rollout 控制变量

四组实验除两个消融轴外必须保持一致：

- 同一个 `pi05_base` 初始化和 full-finetune recipe；
- 同一批成功示范、相同 train/validation 划分；
- 相同 batch size、训练步数、优化器和随机种子；
- 相同 action horizon、实际 execution horizon、query cadence 和 temporal ensemble 设置；
- 相同相机、图像预处理、robot state 和 prompt；
- 相同最终评测 seed 列表。

建议批次名：

```text
pi05_rearrange_state_token_boundary_ablation
```

四个 cell 共用一个 train config：

```text
pi05_rearrange_state_token_boundary_ablation
```

Parallel/Serial 由 `--model.key-state-token-mode` 覆盖，Soft/Hard 由
`--data.hard-action-boundary` 覆盖；四组用不同 `exp_name`，并登记在同一个 W&B group 下。

低成功率下单次 rollout 方差较大。建议分两阶段：

1. screening：四组各固定一个训练 seed，每组 100 条相同评测种子；
2. confirmation：对有差异的组补足 3 个训练 seed，并在独立固定评测种子上复验。

checkpoint 选择规则必须在跑 final evaluation 前写死；不得用同一组 100 条最终评测结果
挑 checkpoint。可以固定训练步数，或用独立 dev seeds 选择。

## 11. 指标与诊断

### 11.1 主指标

- episode success rate；
- 95% binomial confidence interval；
- 四组在同一 rollout seed 上的 paired success difference。

### 11.2 任务分解指标

- `press_cnt == 0 / 1 / >1` 的比例；
- 进入环境 stage 1、stage 2 的比例；
- 按按钮时第一块是否已正确放置；
- 按钮重试次数、无进展循环次数和完成步数；
- P0/P1/P2 phase confusion matrix；
- `button_press_status` precision、recall、切换延迟与提前切换率；
- Serial predicted-state 与 oracle-state 的成功率差。

### 11.3 动作质量指标

- 被 Hard 改写的样本比例和每个 chunk 的 hold-tail 长度；
- action velocity/jerk；
- 按钮处停留时间；
- phase/guard 前后轨迹是否出现明显停顿、反复下压或多次触发。

每个 cell 至少保存前 5 条带 overlay 的 rollout 视频。overlay 显示：预测 phase、按钮
status、是否命中 guard、chunk 内当前位置和 `press_cnt`（`press_cnt` 仅用于评测可视化）。

## 12. 两个必要的非主实验诊断

这两项不进入 2×2 主表，但对解释 Hard 的收益必不可少：

### 12.1 Dwell-only

在按钮底部固定多保持若干 control steps，但不预测确认 token，也不因确认结果改变策略。

- 若 Dwell-only 与 Hard 同样改善，主要问题是接触时间/动作幅度，而不是 phase 确认；
- 若 Hard 明显更好，requery 与状态分支才可能是关键。

### 12.2 Oracle-hard

使用仿真 `press_flag`、按钮关节或事件真值决定何时释放，仅作上界诊断。

- Oracle-hard 好、learned-hard 差：视觉/机器人 proprio 无法可靠判断按钮状态；
- 两者都差：问题更可能是接近轨迹、控制精度或 repeat-last 的动作语义；
- 两者都好：Hard guard 可行，应继续优化 learned state token。

## 13. 主要风险

### 13.1 按钮状态可能不可观测

按钮位移小、相机角度有限，且当前 robot state 不含按钮关节。若单帧无法区分“差一点按到”
和“已经触发”，Serial 结构不会凭空解决感知问题。Oracle-hard 对照用于验证这一点；必要时
再讨论多帧视觉、接触/力觉或环境可提供的非特权传感器，而不是直接把仿真真值喂给正式策略。

### 13.2 Hard 可能降低连贯性

repeat-last 会在示范中制造大量静止尾部，可能让 flow matching loss 被 hold action 主导，
使整体动作变慢或在边界抖动。因此必须统计 hold ratio，并限制首版只作用于按钮 guard。

### 13.3 边界 off-by-one 的代价不对称

- `T` 过早：重复的是未到位动作，仍可能 `press_cnt == 0`；
- `T` 过晚：重复的是释放动作，Hard 语义完全失效；
- 真机中过长保持还可能带来接触力风险。

所以 `T` 必须来自物理触发事件，并在少量轨迹上逐帧可视化审计。

### 13.4 Serial 的 exposure bias

训练只见成功示范和 GT current-state token，部署却会遇到失败图像和错误 predicted token，
所以 Serial 可能比 Parallel 更脆弱。主实验接受该 exposure bias，并用第 6.5 节的
predicted/oracle 双评测量化；scheduled sampling、token dropout 和失败恢复数据留作后续扩展。

### 13.5 Soft 跨 phase 不等于错误

抓取中的张开、接近、合爪、抬起天然适合一个连续 chunk。成功时跨 phase 提高连贯性，失败
后重新观察又能自然重试。本文不把所有 phase boundary 都硬切；Hard 应只用于后果不可逆或
必须基于新观测确认的局部 guard。

## 14. 相关工作定位

文献通常分别研究 action chunk、receding-horizon execution、层级技能或 option termination，
较少把“带语义 phase token 的 chunk 是否允许跨边界”作为单独的数据监督轴。这更像一个
连接模型、数据标注和执行器的系统设计问题，而不只是实现 trick。

- ACT 使用固定 action chunk 与 temporal ensemble：
  [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://www.roboticsproceedings.org/rss19/p016.pdf)
- Diffusion Policy 使用 receding-horizon action sequence：
  [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- options/intra-option 方法显式研究技能与 termination：
  [Intra-Option Learning about Temporally Abstract Actions](https://icml.cc/Conferences/1998/papers/paper162.html)
- RT-H 用语言 motion 层级化 VLA 控制，但不是本文相同的 boundary supervision：
  [RT-H](https://arxiv.org/abs/2403.01823)

## 15. 评审后实施顺序

1. 给 `rearrange_blocks` 补充按钮 press-down/release 的细粒度事件记录；
2. 生成少量轨迹，逐帧审计 `T`、保持动作和视觉可辨识性；
3. 定义 factorized state token 和 query-pair dataset schema；
4. 实现 Parallel/Serial 两种模型结构；
5. 实现 Soft/Hard 两种 action target builder，并加单元测试验证边界索引；
6. 先跑少量 overfit/rollout smoke test；
7. 登记并启动 2×2 screening；
8. 根据结果运行 dwell-only、oracle-hard 和多 seed confirmation。

## 16. 本次评审需要确认的决定

1. 是否接受“高层 phase 仍为 3 类，另加 `button_press_status` guard token”的拆分；
2. 是否同意首轮 Hard 只作用于按钮 press-down guard，而不是所有 phase boundary；
3. 若现有数据无法恢复准确的 press-down 事件，是否接受修改环境记录后重新生成/采集数据；
4. Serial 首版采用第 6.4 节定义的一次 block-causal teacher forcing，不同时加入 scheduled
   sampling、soft embedding、straight-through estimator 或 token dropout；
5. 2×2 screening 后，再根据结果决定是否投入变长 horizon 和更多任务。
