# Eval Diagnostics

评测诊断采用“通用记录器 + 任务级条件”的结构，避免在 `script/eval_policy.py`
里硬编码任务成功条件。

每个 eval run 除原有文件外生成：

```text
episode_diagnostics.jsonl  # 每个 episode 的完整条件、事件和主失败原因
diagnostics_summary.json   # 通用聚合后的失败分布、条件满足率和数值统计
```

`_result.txt` 保存便于阅读的失败原因和条件率摘要；W&B 保存相同指标的扁平版本。

## 任务接口

`Base_Task.get_eval_diagnostics(success)` 提供默认 schema。新任务只需 override
这个方法，不需要修改 evaluator：

```python
def get_eval_diagnostics(self, success):
    diagnostics = super().get_eval_diagnostics(success)
    diagnostics["primary_failure_reason"] = self._classify_failure(success)
    diagnostics["conditions"] = {
        "object_on_target": bool(...),
        "gripper_open": bool(...),
    }
    diagnostics["metrics"].update({
        "attempt_count": int(...),
        "target_distance": float(...),
    })
    return diagnostics
```

公共字段语义：

```text
primary_failure_reason  每个失败 episode 一个互斥标签，用于失败分布。
conditions              非互斥布尔条件；None 表示该 episode 不适用，不进入分母。
metrics                 有限数值；聚合为 count / mean / min / max。
events                  关键瞬时事件。用 _record_eval_diagnostic_event() 记录。
```

如果成功条件依赖某个瞬时事件，例如“按下按钮时物体必须已经摆好”，任务应在事件
发生时保存条件快照。不能只依赖终止帧，否则无法区分动作顺序错误和终态被后续动作
改变。

## Rearrange Blocks

`rearrange_blocks` 当前记录：

```text
first_placement_ready
button_pressed（含首次按下时的条件快照）
stage 0 -> 1 / stage 1 -> 2
second_placement_ready
button joint 的 episode 最小值
```

按钮最小值用于区分没有按按钮和按压深度不足。任务成功阈值保持原定义
`button_joint < -0.005`；`-0.001` 用作“发生过不足按压”的诊断阈值，不影响环境
成功判定。
