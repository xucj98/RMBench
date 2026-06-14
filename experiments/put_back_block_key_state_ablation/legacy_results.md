# Key-State 历史结果

本文记录不作为当前正式指标的历史结果，避免以后误用。

## State Input Bug 修复前

下面 3 个 50-rollout 结果来自 state input bug 修复前，模型实际没有收到 32 维 key-state policy state，因此不作为当前 key-state 指标。

| Variant | Result | Source |
| --- | ---: | --- |
| `default` | 0/50 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem/2026-06-03 15:29:15/_result.txt` |
| `mat_hash_p50` | 0/50 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_mat_hash_p50_mem/2026-06-03 15:29:15/_result.txt` |
| `phase_jitter5` | 0/50 | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_phase_jitter5_mem/2026-06-03 15:29:14/_result.txt` |

2026-06-09 的非 overlay 100-rollout rerun 也存在：`default` 和 `mat_hash_p50` 为 0/100，`phase_jitter5` 为 1/100。这组结果没有 key-state overlay，也早于后续 state input 修复，不作为当前结果。

## 中断或仅调试目录

名字里有 `video5_rerun` 但没有 `100rollout` 的中断目录，以及名字里有 `50rollout_video5_rerun` 的中断目录，都不要作为结果使用。

## Baseline 备查

用于主实验对照的普通 pi0 baseline：

```text
put_back_block pi0 LoRA baseline: 4/50 = 0.08
source: eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block/2026-06-01 16:27:39/_result.txt
```
