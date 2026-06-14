# Key-State 调试记录

本文只记录排查过程，不作为主实验结果表。

## State Input Bug

早期 key-state eval 出现 0% 成功率。排查发现，部署推理时虽然生成了 32 维 key-state policy state，但实际传给 policy 的仍是原始 14 维 state，导致模型没有真正收到 phase/mat memory。

该问题已在 `a6737e5` 修复。修复后，`default` variant 的 100-rollout raw eval 达到 55/100。

## Schema Latch

为排查 overlay 中 key-state 中途跳变的问题，曾增加可选的 `key_state_update_mode=schema_latch`。该模式让 phase 单步单调推进，并让 mat 首次变成非 unknown 后锁存。

修复 state input bug 后，对 `default` variant 做过 100-rollout 对照：

| Run | key_state_update_mode | Success |
| --- | --- | ---: |
| `statefix_raw_100rollout_video5` | `raw` | 55/100 |
| `statefix_schema_latch_100rollout_video5` | `schema_latch` | 55/100 |

两组 episode 结果逐条一致。因此主实验使用默认 `raw`，`schema_latch` 只保留为调试工具。

## 按钮阶段失败

为解释 default 仍只有约 55% 成功率，曾做过一次临时 debug 统计：`schema_latch`、20 rollouts、无视频，结果为 11/20。9 个失败 episode 全部满足：

```text
press_cnt = 0
stage_id 从未离开 0
block 曾到达 center
block 曾满足最终目标区域几何条件
```

这说明部分失败不是最终放置位置本身没到，而是环境没有记录到按钮 press，导致 `stage_id` 没有打开最终成功 gate。相关 debug 代码只用于本地排查，不作为正式实验代码提交。

## Failure-Case 视频

2026-06-10 曾用 `schema_latch` 跑过一次 5-video 调试，启动后在看到 failure case 后手动停止：

```text
ckpt_setting: pi0_put_back_block_key_state_default_mem_schema_latch_video5
train_config_name: pi0_aloha_put_back_block_key_state_default_lora
model_name: pi0_put_back_block_key_state_default
checkpoint_id: 30000
key_state_update_mode: schema_latch
test_num: 5
eval_video_log: true
eval_video_count: 5
eval_video_key_state_overlay: true
result_dir: eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_schema_latch_video5/2026-06-10 21:54:17
```

该运行是调试性视频排查，不作为正式 eval 指标。
