# RMBench Experiments Index

## 任务级主表

空白表示当前没有正式评测结果。除特别说明外，复现结果均采用 100-rollout eval；
`pi0_lora_key_state` 使用默认 30k checkpoint 结果。论文 DP / Mem-0 / Pi0.5
来自 `PROGRESS.md` 记录的 RMBench Table 1。

| Task | Paper DP | Paper Mem-0 | Paper Pi0.5 | Repro DP | dp_key_state | Repro Mem-0 | pi0_lora | pi0_full | pi05_full | pi0_lora_key_state | pi0_full_key_state | pi05_full_key_state @20k | @30k | @40k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `observe_and_pickup` | 1% | 4% | 9% | 2% |  | 4% | 4% |  |  |  |  |  |  |  |
| `rearrange_blocks` | 0% | 89% | 13% | 0% | running | 0% | 1% | 21% | 20% | 3% | 37% | 30% | 44% | 41% |
| `put_back_block` | 0% | 90% | 11% | 0% | running |  | 7% |  |  | 55% | 68% | 65% | 59% | 60% |
| `swap_blocks` | 11% | 67% | 24% | 15% | running |  | 16% |  | 14% | 44% |  | 86% | 84% | 93% |
| `swap_T` | 20% | 14% | 15% | 11% |  |  | 13% |  |  |  |  |  |  |  |
| `battery_try` | 10% | 28% | 16% | 13% | running |  | 8% | 13% | 17% | 15% |  | 33% | 32% | 34% |
| `blocks_ranking_try` | 10% | 18% | 6% | 3% |  |  | 16% |  |  |  |  |  |  |  |
| `cover_blocks` | 0% | 68% | 0% | 0% | running |  | 1% |  |  | 0% |  | 10% | 23% | 15% |
| `press_button` | 0% | 0% | 0% | 0% |  |  | 3% |  |  |  |  |  |  |  |

## 实验索引

Markdown 标准表格和 GitHub Flavored Markdown 都不支持真正的单元格合并。
这里统一用普通 Markdown 表格：一行表示一个 train/eval 组合；同一个
`exp group`、`exp name` 或 checkpoint 对应多个 eval 条件时，重复填写对应单元格。

本表只维护正式实验和正在运行的正式实验；smoke、启动测试和早期调试结果不进入总表。
路径为仓库相对路径。完整命令、commit 和环境快照以 checkpoint / eval result 目录内的
metadata、`command.txt`、`config.yaml` 以及 W&B 记录为准。

| exp group | exp name | success rate | train state | eval state | ckpt path | eval result path |
| --- | --- | ---: | --- | --- | --- | --- |
| `dp_reproduction` | `observe_and_pickup` | 2/100 = 2% | finished | finished | `policy/DP/checkpoints/observe_and_pickup-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/observe_and_pickup` |
| `dp_reproduction` | `put_back_block` | 0/100 = 0% | finished | finished | `policy/DP/checkpoints/put_back_block-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/put_back_block` |
| `dp_reproduction` | `rearrange_blocks` | 0/100 = 0% | finished | finished | `policy/DP/checkpoints/rearrange_blocks-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/rearrange_blocks` |
| `dp_reproduction` | `swap_T` | 11/100 = 11% | finished | finished | `policy/DP/checkpoints/swap_T-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/swap_T` |
| `dp_reproduction` | `swap_blocks` | 15/100 = 15% | finished | finished | `policy/DP/checkpoints/swap_blocks-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/swap_blocks` |
| `dp_reproduction` | `cover_blocks` | 0/100 = 0% | finished | finished | `policy/DP/checkpoints/cover_blocks-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/cover_blocks` |
| `dp_reproduction` | `battery_try` | 13/100 = 13% | finished | finished | `policy/DP/checkpoints/battery_try-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/battery_try` |
| `dp_reproduction` | `press_button` | 0/100 = 0% | finished | finished | `policy/DP/checkpoints/press_button-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/press_button` |
| `dp_reproduction` | `blocks_ranking_try` | 3/100 = 3% | finished | finished | `policy/DP/checkpoints/blocks_ranking_try-demo_clean-50-0/600.ckpt` | `eval_result/dp_reproduction/blocks_ranking_try` |
|  |  |  |  |  |  |  |
| `dp_key_state` | `put_back_block` | - | running | todo | `policy/DP/checkpoints/put_back_block-demo_clean_state_key_state-50-0/600.ckpt` | - |
| `dp_key_state` | `rearrange_blocks` | - | running | todo | `policy/DP/checkpoints/rearrange_blocks-demo_clean_state_key_state-50-0/600.ckpt` | - |
| `dp_key_state` | `swap_blocks` | - | running | todo | `policy/DP/checkpoints/swap_blocks-demo_clean_state_key_state-50-0/600.ckpt` | - |
| `dp_key_state` | `battery_try` | - | running | todo | `policy/DP/checkpoints/battery_try-demo_clean_state_key_state-50-0/600.ckpt` | - |
| `dp_key_state` | `cover_blocks` | - | running | todo | `policy/DP/checkpoints/cover_blocks-demo_clean_state_key_state-50-0/600.ckpt` | - |
|  |  |  |  |  |  |  |
| `pi0_lora_baseline` | `swap_blocks` | 16/100 = 16% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/swap_blocks/30000` | `eval_result/pi0_lora_baseline/swap_blocks_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `swap_T` | 13/100 = 13% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/swap_T/30000` | `eval_result/pi0_lora_baseline/swap_T_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `put_back_block` | 7/100 = 7% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/put_back_block/30000` | `eval_result/pi0_lora_baseline/put_back_block_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `observe_and_pickup` | 4/100 = 4% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/observe_and_pickup/30000` | `eval_result/pi0_lora_baseline/observe_and_pickup_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `rearrange_blocks` | 1/100 = 1% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/rearrange_blocks/30000` | `eval_result/pi0_lora_baseline/rearrange_blocks_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `cover_blocks` | 1/100 = 1% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/cover_blocks/30000` | `eval_result/pi0_lora_baseline/cover_blocks_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `battery_try` | 8/100 = 8% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/battery_try/30000` | `eval_result/pi0_lora_baseline/battery_try_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `press_button` | 3/100 = 3% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/press_button/30000` | `eval_result/pi0_lora_baseline/press_button_raw_100_video5_20260614_pi0_baseline_100` |
| `pi0_lora_baseline` | `blocks_ranking_try` | 16/100 = 16% | finished | finished | `policy/pi05/checkpoints/pi0_lora_baseline/blocks_ranking_try/30000` | `eval_result/pi0_lora_baseline/blocks_ranking_try_raw_100_video5_20260614_pi0_baseline_100` |
|  |  |  |  |  |  |  |
| `pi0_key_state_baseline` | `put_back_block@ckpt30000` | 55/100 = 55% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_lora/pi0_put_back_block_key_state_default/30000` | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_raw_100rollout_video5/2026-06-10 22:41:07` |
| `pi0_key_state_baseline` | `rearrange_blocks@ckpt20000` | 15/100 = 15% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_rearrange_blocks/20000` | `eval_result/pi0_key_state_baseline/rearrange_blocks_ckpt20000` |
| `pi0_key_state_baseline` | `rearrange_blocks@ckpt30000` | 3/100 = 3% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_rearrange_blocks/30000` | `eval_result/pi0_key_state_baseline/rearrange_blocks` |
| `pi0_key_state_baseline` | `swap_blocks@ckpt20000` | 40/100 = 40% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_swap_blocks/20000` | `eval_result/pi0_key_state_baseline/swap_blocks_ckpt20000` |
| `pi0_key_state_baseline` | `swap_blocks@ckpt30000` | 44/100 = 44% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_swap_blocks/30000` | `eval_result/pi0_key_state_baseline/swap_blocks` |
| `pi0_key_state_baseline` | `battery_try@ckpt20000` | 10/100 = 10% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_battery_try/20000` | `eval_result/pi0_key_state_baseline/battery_try_ckpt20000` |
| `pi0_key_state_baseline` | `battery_try@ckpt30000` | 15/100 = 15% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_battery_try/30000` | `eval_result/pi0_key_state_baseline/battery_try` |
| `pi0_key_state_baseline` | `battery_try@ckpt30000_pi0step20` | 14/100 = 14% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_battery_try/30000` | `eval_result/pi0_key_state_baseline/battery_try_ckpt30000_pi0step20` |
| `pi0_key_state_baseline` | `cover_blocks@ckpt20000` | 0/100 = 0% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_cover_blocks/20000` | `eval_result/pi0_key_state_baseline/cover_blocks_ckpt20000` |
| `pi0_key_state_baseline` | `cover_blocks@ckpt30000` | 0/100 = 0% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_baseline_cover_blocks/30000` | `eval_result/pi0_key_state_baseline/cover_blocks` |
|  |  |  |  |  |  |  |
| `put_back_block_key_state_ablation` | `default@50rollout_raw` | 27/50 = 54% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_lora/pi0_put_back_block_key_state_default/30000` | `eval_result/put_back_block_key_state_ablation/default_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `mat_hash_p50` | 21/50 = 42% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_mat_hash_p50_lora/pi0_put_back_block_key_state_mat_hash_p50/30000` | `eval_result/put_back_block_key_state_ablation/mat_hash_p50_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `wmat_margin10` | 19/50 = 38% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_wmat_margin10_lora/pi0_put_back_block_key_state_wmat_margin10/30000` | `eval_result/put_back_block_key_state_ablation/wmat_margin10_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `phase_jitter5` | 19/50 = 38% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_phase_jitter5_lora/pi0_put_back_block_key_state_phase_jitter5/30000` | `eval_result/put_back_block_key_state_ablation/phase_jitter5_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `phase_lag10` | 18/50 = 36% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_phase_lag10_lora/pi0_put_back_block_key_state_phase_lag10/30000` | `eval_result/put_back_block_key_state_ablation/phase_lag10_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `mat_first` | 16/50 = 32% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_mat_first_lora/pi0_put_back_block_key_state_mat_first/30000` | `eval_result/put_back_block_key_state_ablation/mat_first_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `wmat_margin20` | 15/50 = 30% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_wmat_margin20_lora/pi0_put_back_block_key_state_wmat_margin20/30000` | `eval_result/put_back_block_key_state_ablation/wmat_margin20_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `phase_lag20` | 9/50 = 18% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_phase_lag20_lora/pi0_put_back_block_key_state_phase_lag20/30000` | `eval_result/put_back_block_key_state_ablation/phase_lag20_raw_50_video2_20260611_214744` |
| `put_back_block_key_state_ablation` | `default@100rollout_raw` | 55/100 = 55% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_lora/pi0_put_back_block_key_state_default/30000` | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_raw_100rollout_video5/2026-06-10 22:41:07` |
| `put_back_block_key_state_ablation` | `default@100rollout_schema_latch` | 55/100 = 55% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_lora/pi0_put_back_block_key_state_default/30000` | `eval_result/put_back_block/pi05/demo_clean_eval/pi0_put_back_block_key_state_default_mem_statefix_schema_latch_100rollout_video5/2026-06-10 22:41:07` |
| `put_back_block_key_state_ablation` | `default_full_b32` | 68/100 = 68% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_put_back_block_key_state_default_full_b32/pi0_put_back_block_key_state_default_full_b32/30000` | `eval_result/put_back_block_key_state_ablation/default_full_b32_raw_100_video5_20260614_170220` |
|  |  |  |  |  |  |  |
| `pi0_full_baseline` | `rearrange_blocks` | 21/100 = 21% | finished | finished | `policy/pi05/checkpoints/pi0_full_baseline/pi0_full_baseline_rearrange_blocks/30000` | `eval_result/pi0_full_baseline/rearrange_blocks` |
| `pi0_full_baseline` | `battery_try` | 13/100 = 13% | finished | finished | `policy/pi05/checkpoints/pi0_full_baseline/pi0_full_baseline_battery_try/30000` | `eval_result/pi0_full_baseline/battery_try` |
|  |  |  |  |  |  |  |
| `pi05_full_baseline` | `rearrange_blocks` | 20/100 = 20% | finished | finished | `policy/pi05/checkpoints/pi05_full_baseline/pi05_full_baseline_rearrange_blocks/20000` | `eval_result/pi05_full_baseline/rearrange_blocks` |
| `pi05_full_baseline` | `battery_try` | 17/100 = 17% | finished | finished | `policy/pi05/checkpoints/pi05_full_baseline/pi05_full_baseline_battery_try/20000` | `eval_result/pi05_full_baseline/battery_try` |
|  |  |  |  |  |  |  |
| `pi0_full_key_state` | `rearrange_blocks` | 37/100 = 37% | finished | finished | `policy/pi05/checkpoints/pi0_full_key_state/pi0_full_key_state_rearrange_blocks/30000` | `eval_result/pi0_full_key_state/rearrange_blocks` |
|  |  |  |  |  |  |  |
| `pi05_full_key_state` | `legacy_rearrange_blocks@ckpt30000` | 44/100 = 44% | finished | finished | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_rearrange_blocks/30000` | `eval_result/pi05_full_key_state/rearrange_blocks` |
| `pi05_full_key_state` | `legacy_swap_blocks@ckpt20000` | 85/100 = 85% | finished | finished | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_swap_blocks/20000` | `eval_result/pi05_full_key_state/swap_blocks` |
| `pi05_full_key_state` | `legacy_battery_try@ckpt20000` | 29/100 = 29% | finished | finished | `policy/pi05/checkpoints/full_key_state/pi05_full_key_state_battery_try/20000` | `eval_result/pi05_full_key_state/battery_try` |
| `pi05_full_key_state` | `put_back_block@ckpt20k` | 65/100 = 65% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/put_back_block/20000` | `eval_result/pi05_full_key_state/put_back_block@ckpt20k` |
| `pi05_full_key_state` | `put_back_block@ckpt30k` | 59/100 = 59% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/put_back_block/30000` | `eval_result/pi05_full_key_state/put_back_block@ckpt30k` |
| `pi05_full_key_state` | `put_back_block@ckpt40k` | 60/100 = 60% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/put_back_block/40000` | `eval_result/pi05_full_key_state/put_back_block@ckpt40k` |
| `pi05_full_key_state` | `rearrange_blocks@ckpt20k` | 30/100 = 30% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/rearrange_blocks/20000` | `eval_result/pi05_full_key_state/rearrange_blocks@ckpt20k` |
| `pi05_full_key_state` | `rearrange_blocks@ckpt30k` | 44/100 = 44% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/rearrange_blocks/30000` | `eval_result/pi05_full_key_state/rearrange_blocks@ckpt30k` |
| `pi05_full_key_state` | `rearrange_blocks@ckpt40k` | 41/100 = 41% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/rearrange_blocks/40000` | `eval_result/pi05_full_key_state/rearrange_blocks@ckpt40k` |
| `pi05_full_key_state` | `swap_blocks@ckpt20k` | 86/100 = 86% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/swap_blocks/20000` | `eval_result/pi05_full_key_state/swap_blocks@ckpt20k` |
| `pi05_full_key_state` | `swap_blocks@ckpt30k` | 84/100 = 84% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/swap_blocks/30000` | `eval_result/pi05_full_key_state/swap_blocks@ckpt30k` |
| `pi05_full_key_state` | `swap_blocks@ckpt40k` | 93/100 = 93% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/swap_blocks/40000` | `eval_result/pi05_full_key_state/swap_blocks@ckpt40k` |
| `pi05_full_key_state` | `battery_try@ckpt20k` | 33/100 = 33% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/battery_try/20000` | `eval_result/pi05_full_key_state/battery_try@ckpt20k` |
| `pi05_full_key_state` | `battery_try@ckpt30k` | 32/100 = 32% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/battery_try/30000` | `eval_result/pi05_full_key_state/battery_try@ckpt30k` |
| `pi05_full_key_state` | `battery_try@ckpt40k` | 34/100 = 34% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/battery_try/40000` | `eval_result/pi05_full_key_state/battery_try@ckpt40k` |
| `pi05_full_key_state` | `cover_blocks@ckpt20k` | 10/100 = 10% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/cover_blocks/20000` | `eval_result/pi05_full_key_state/cover_blocks@ckpt20k` |
| `pi05_full_key_state` | `cover_blocks@ckpt30k` | 23/100 = 23% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/cover_blocks/30000` | `eval_result/pi05_full_key_state/cover_blocks@ckpt30k` |
| `pi05_full_key_state` | `cover_blocks@ckpt40k` | 15/100 = 15% | finished | finished | `policy/pi05/checkpoints/pi05_full_key_state/cover_blocks/40000` | `eval_result/pi05_full_key_state/cover_blocks@ckpt40k` |
|  |  |  |  |  |  |  |
| `pi0_key_state_encoding_ablation` | `cover_blocks_label_id` | 0/100 = 0% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_encoding_ablation_cover_blocks_label_id/30000` | `eval_result/pi0_key_state_encoding_ablation/cover_blocks_label_id` |
| `pi0_key_state_encoding_ablation` | `battery_try_micro_stage_label_id` | 13/100 = 13% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/pi0_key_state_encoding_ablation_battery_try_micro_stage_label_id/30000` | `eval_result/pi0_key_state_encoding_ablation/battery_try_micro_stage_label_id` |
|  |  |  |  |  |  |  |
| `cover_blocks_key_state_design` | `exec2_attr3_no_phase` | 0/100 = 0% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/cover_blocks_key_state_design_exec2_attr3_no_phase/30000` | `eval_result/cover_blocks_key_state_design/exec2_attr3_no_phase` |
| `cover_blocks_key_state_design` | `phase_exec2_attr3` | 1/100 = 1% | finished | finished | `policy/pi05/checkpoints/pi0_aloha_key_state_lora/cover_blocks_key_state_design_phase_exec2_attr3/30000` | `eval_result/cover_blocks_key_state_design/phase_exec2_attr3` |
