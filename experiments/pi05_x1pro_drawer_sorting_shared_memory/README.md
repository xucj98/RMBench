# Pi0.5 X1Pro Drawer Sorting Shared-Memory Training

本实验组比较同一语义 memory 的两种表示：Full State dense one-hot 与 Serial Soft state
token。逐 timestamp 定义见
[`docs/x1pro_drawer_sorting_state_transition_design.md`](../../docs/x1pro_drawer_sorting_state_transition_design.md)。

## v2：S2M + 层进度（当前设计）

v2 共享两个 memory 字段：

```text
completed_layers ∈ {completed_0, completed_1, completed_2, completed_3}
drawer_target    ∈ {observe, item_1, item_2, item_3}
```

- 原始/训练频率：30 Hz / 15 Hz；
- robot：14D slave state → 14D master action（S2M）；
- history/current/future：0/1/0；
- action horizon：30；
- soft action boundary；
- batch size：32，steps：30,000，seed：42；
- 数据集：`drawer_sorting_x1pro_shared_memory_s2m_15hz_v2`；
- W&B group：`pi05_x1pro_drawer_sorting_shared_memory`。

Full State 与 Serial Soft 共用同一个 LeRobot 数据集、schema 和逐帧 sidecar，只通过各自
adapter 选择 dense memory 或离散 token。

转换命令：

```bash
cd policy/pi05
uv run python examples/x2robot/convert_drawer_sorting_to_lerobot.py \
  --config ../../converter_configs/shared_memory/drawer_sorting_x1pro.yaml \
  --num-workers 20
```

训练 manifest：`jobs_train_s2m_v2.json`。

| run | config | checkpoint | 状态 |
| --- | --- | --- | --- |
| `s2m_full_state_v2_seed42` | `pi05_x1pro_drawer_sorting_s2m_full_state` | `policy/pi05/checkpoints/pi05_x1pro_drawer_sorting_s2m_full_state/s2m_full_state_v2_seed42/30000` | 等待数据转换/训练 |
| `s2m_serial_soft_v2_seed42` | `pi05_x1pro_drawer_sorting_s2m_serial_soft` | `policy/pi05/checkpoints/pi05_x1pro_drawer_sorting_s2m_serial_soft/s2m_serial_soft_v2_seed42/30000` | 等待数据转换/训练 |

## v1：单字段 SM2SM（已被 v2 取代，保留复现）

v1 只有 `drawer_target`，采用 28D SM2SM、3/1/3 history/current/future 和 latency 3。
由于无法表达已完成第几层，它不再用于当前设计，但数据、配置和 checkpoint 均保留。

| run | config | checkpoint | 状态 |
| --- | --- | --- | --- |
| `full_state_seed42` | `pi05_x1pro_drawer_sorting_full_state` | `policy/pi05/checkpoints/pi05_x1pro_drawer_sorting_full_state/full_state_seed42/30000` | 30k 完成，W&B `qcikknti`；v1 superseded |
| `serial_soft_seed42` | `pi05_x1pro_drawer_sorting_serial_soft` | `policy/pi05/checkpoints/pi05_x1pro_drawer_sorting_serial_soft/serial_soft_seed42/30000` | 30k 完成，W&B `zcqsxzjk`；v1 superseded |

v1 manifest 是 `jobs_train.json`，数据集是
`drawer_sorting_x1pro_shared_memory_sm2sm_15hz`。v1 正式训练于 2026-08-22 启动，两个 run
分别于 2026-08-24 03:43 和 04:03（Asia/Shanghai）正常结束，return code 均为 0。

## 数据审计与存放

- 原始目录：`data/x1pro_table_clean`（指向 `/mnt/public3/datasets/x1pro/table_clean`）；
- `anno/tags.json` 含 `drawer_sorting` 的 episode：134；
- 严格可用：119；跳过：15；
- 有效 episode 必须恰好含三个执行区间，4/5/6 各一次；观察 1/2/3 允许缺失；
- LeRobot cache：`/root/.cache/huggingface/lerobot`，该路径本身已软链到共享存储。

不会创建额外的 `/mnt/public3/xcj/lerobot_datasets` 数据根目录。
