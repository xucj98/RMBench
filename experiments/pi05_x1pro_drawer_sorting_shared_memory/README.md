# Pi0.5 X1Pro Drawer Sorting Shared-Memory Training

本批次使用真机 X1Pro `table_clean` 数据训练两种共享 memory 表示：

- `full_state`：3D implicit-unknown one-hot memory 注入连续 state/action；
- `serial_soft`：先预测离散 `drawer_target`，再用 GT token teacher forcing 条件化 action。

两者共享同一个 LeRobot 数据集和同一个语义 schema。逐 timestamp 状态定义、缺观察片段、
4/5/6 execution override 以及 Full State memory action loss mask 详见
`docs/x1pro_drawer_sorting_state_transition_design.md`。

## 数据

- 原始目录：`data/x1pro_table_clean`（本机软链，不写入代码配置的绝对 `/mnt` 路径）；
- 选择：`anno/tags.json` 包含 `drawer_sorting`；
- 标注：`anno/sort.json` 的 1/2/3 为观察，4/5/6 为对应 item 抓放；
- 原始/训练频率：30 Hz / 15 Hz；
- 严格可用：119 episodes，跳过 15 episodes；
- LeRobot repo：`drawer_sorting_x1pro_shared_memory_sm2sm_15hz`；
- cache：`/root/.cache/huggingface/lerobot`，其本身是共享存储软链，不另建数据根目录。

转换命令：

```bash
cd policy/pi05
uv run python examples/x2robot/convert_drawer_sorting_to_lerobot.py \
  --config ../../converter_configs/shared_memory/drawer_sorting_x1pro.yaml \
  --num-workers 20 --overwrite
```

## 训练设置

- action horizon：30（15 Hz 下 2 秒）；
- state history/current/future：3/1/3；
- SM2SM robot state/action：28D；
- batch size：32；
- steps：30,000；
- seed：42；
- base checkpoint：Pi0.5 base；
- W&B group：`pi05_x1pro_drawer_sorting_shared_memory`；
- 代码提交：`27290afb7c151cc2e1009ac3c2c6ae6ee5e6ee4f`。

正式训练 manifest：`jobs_train.json`。计划使用 GPU1/2 各运行一个模型。

| run | config | checkpoint | 状态 |
| --- | --- | --- | --- |
| `full_state_seed42` | `pi05_x1pro_drawer_sorting_full_state` | `policy/pi05/checkpoints/pi05_x1pro_drawer_sorting_full_state/full_state_seed42/30000` | 等待数据转换与 norm stats |
| `serial_soft_seed42` | `pi05_x1pro_drawer_sorting_serial_soft` | `policy/pi05/checkpoints/pi05_x1pro_drawer_sorting_serial_soft/serial_soft_seed42/30000` | 等待数据转换与 norm stats |
