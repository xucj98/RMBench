# DM0.5 on `swap_blocks`

This experiment fine-tunes the open-source Dexmal `DM05-robotwin2` checkpoint
on the 50 RMBench `swap_blocks` demonstrations and evaluates it in
`demo_clean_eval`. Training is intentionally limited to four GPUs. The
RoboTwin2 checkpoint supplies a stronger ALOHA/RoboTwin initialization while
remaining a DM0.5 full fine-tune.

The vendored OpenDM source is upstream commit
`e89fcbaa0408ca0fb04a410bfab50cc15eb73fdb`.

## Four-A800 feasibility

The upstream RoboTwin2 recipe recommends eight GPUs and uses FSDP
`SHARD_GRAD_OP`. On this host, four A800 80GB GPUs can full-fine-tune DM0.5
when using FSDP `FULL_SHARD` (the default in this experiment):

- 20-step BF16/SDPA smoke run, per-device batch 1: 60.7--65.3 GiB/GPU,
  stable steps about 1.7 seconds after warm-up.
- 10-step BF16/SDPA smoke run, per-device batch 2: about 76.8 GiB/GPU,
  stable steps about 2.6 seconds after warm-up.
- GPUs 4--7 remained unused throughout both four-GPU runs.
- FlashAttention 2, per-device batch 2 (formal setting): 60.9--66.8
  GiB/GPU after warm-up, stable steps about 2.25 seconds.
- FlashAttention 2, per-device batch 3/4: about 79.0 GiB/GPU, tested but
  rejected for the formal run because the safety margin is too small.

FlashAttention 2 is used for the formal run. SDPA remains a tested fallback;
batch 1 is the conservative fallback setting.

The formal configuration uses four GPUs, BF16, FSDP `FULL_SHARD`,
FlashAttention 2, per-device batch 2 (global batch 8), and a 10,000-step
upper bound. Checkpoints are saved every 2,500 steps. Validation selected
step 5,000 as the best checkpoint and stopped the run at step 10,000 because
both later validation points regressed.

## Data

Generate lightweight OpenDM JSONL indices. Camera JPEGs remain in the existing
HDF5 files and are loaded directly during training. The loader repairs the
red/blue channel swap introduced by RMBench's OpenCV HDF5 processing path;
without that repair, red blocks in evaluation appear blue in every training
frame. The 50 episodes contain 30,017 frames in total.

```bash
policy/DM05/.venv/bin/python \
  policy/DM05/scripts/convert_rmbench_hdf5_to_jsonl.py \
  --source-dir policy/pi05/processed_data/swap_blocks-demo_clean-50 \
  --output-dir policy/DM05/data/rmbench/swap_blocks \
  --prompt-file description/task_instruction/swap_blocks.json
```

## Train (4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  policy/DM05/scripts/train_rmbench_swap_blocks.sh
```

For an end-to-end smoke run, set `DM05_TRAIN_STEPS=2`,
`DM05_SAVE_STEPS=2`, and `DM05_BATCH_SIZE=1`.

## Evaluate

Start the OpenDM service on one GPU, then run the RMBench evaluator from the
`RMBench` Conda environment on another GPU:

```bash
CUDA_VISIBLE_DEVICES=0 policy/DM05/scripts/serve_rmbench_swap_blocks.sh \
  /mnt/public3/xcj/rmbench/dm05/user_checkpoints/swap_blocks/best-checkpoint-5000

CUDA_VISIBLE_DEVICES=4 /root/miniconda3/envs/RMBench/bin/python \
  script/eval_policy.py --config policy/DM05/deploy_policy.yml --overrides \
  --task_name swap_blocks --task_config demo_clean_eval \
  --ckpt_setting dm05_swap_blocks_rgb_best5000_eval100 --seed 0 \
  --policy_name DM05 --test_num 100
```

## Measured result

Checkpoint selection used the same five sanity seeds at each save point:

- step 2,500: 0/5;
- step 5,000: 1/5, followed by 5/20 (25.0%) on a separate 20-episode run;
- step 7,500: 0/5;
- step 10,000: 0/5.

The preserved step-5,000 checkpoint was then evaluated without parameter or
seed filtering on 100 `demo_clean_eval` episodes (`seed=0`): **10/100,
10.0% success**. The evaluator wrote 100 episode log records and 100 episode
diagnostic records. Results are in:

```text
eval_result/swap_blocks/DM05/demo_clean_eval/
  dm05_swap_blocks_rgb_best5000_eval100/2026-09-02 02:41:01/
```

The official unmodified `DM05-robotwin2` checkpoint and the initial
incorrect-color step-5,000 run both scored 0/5 on the same sanity setup. These
numbers are specific to this RMBench task configuration and should not be
confused with the aggregate RoboTwin2 result reported by OpenDM.
