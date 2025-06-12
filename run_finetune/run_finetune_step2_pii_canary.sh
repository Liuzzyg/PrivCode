#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

# Script settings
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
# MODEL_PATH="ise-uiuc/Magicoder-S-DS-6.7B"

MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base" "Qwen/Qwen2.5-Coder-7B" "Qwen/CodeQwen1.5-7B" "google/codegemma-7b")

# MODEL_PATHS=( "google/codegemma-7b")

# Training settings
MAX_STEPS=110
# MAX_STEPS=10000
BATCH_SIZE=2
GRAD_ACCUM_STEPS=16

# LAMBDA=1000
# KL_STEP=10000

# the bigger, the more fastly lambda declines
ALPHA=0.01  # main

# ALPHA=0.2  # for ablation
# ALPHA=0.0001  # for ablation

MAX_LAMBDAs=(100)
MIN_LAMBDA=0.1

REPs=(1 10 100)
# REPs=( 100 )

# DP settings
TARGET_EPSILONs=(4 )
NON_PRIVATE="no"  # Set to "y" for non-private training
# NON_PRIVATE="y"

# Misc settings
LOG_FREQ=1
SAVE_FREQ=10
# SAVE_FREQ_EPOCH=1



for REP in "${REPs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for TARGET_EPSILON in "${TARGET_EPSILONs[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
                if [[ "$NON_PRIVATE" == "y" || "$NON_PRIVATE" == "yes" ]]; then
                    DATASET_NAME="canary/origin_data/pii_instruction_dataset_canary_rep${REP}.jsonl"

                    OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/canary/${MODEL_NAME}/dpinf_baseline_rep${REP}"
                else
                    DATASET_NAME="data/canary/privsyn/Qwen2.5-Coder-1.5B/codeonly/private_syndata_55k_dp4_lambda100to0.1_alpha0.01_datasize55500_rep${REP}_modified.jsonl"

                    OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/canary/${MODEL_NAME}/dp4_rep${REP}"
                fi

                # OUTPUT_DIR="z_results"

                # Run the finetune script using deepspeed
                deepspeed finetune_step2.py \
                    --model_path $MODEL_PATH \
                    --dataset_name $DATASET_NAME \
                    --max_steps $MAX_STEPS \
                    --batch_size $BATCH_SIZE \
                    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
                    --output_dir $OUTPUT_DIR \
                    --log_freq $LOG_FREQ \
                    --save_freq $SAVE_FREQ \
                    --target_epsilon $TARGET_EPSILON \
                    --non_private $NON_PRIVATE \
                    --deepspeed_config examples/codegen/finetune/config_stage2.json
            done
        done
    done
done


exit 0
