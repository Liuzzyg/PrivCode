#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0

# Script settings
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"

DATASET_NAME="ise-uiuc/Magicoder-OSS-Instruct-75K"

# Training settings
MAX_STEPS=200
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILONs=(4 10 1)
NON_PRIVATE="no"  # Set to "y" for non-private training
# NON_PRIVATE="y"

# Misc settings
LOG_FREQ=1
SAVE_FREQ=10
# SAVE_FREQ_EPOCH=1

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

for TARGET_EPSILON in "${TARGET_EPSILONs[@]}"; do

    if [[ "$NON_PRIVATE" == "y" || "$NON_PRIVATE" == "yes" ]]; then
        OUTPUT_DIR=".../checkpoints_codeonly/magicoder/${MODEL_NAME}/dpinf_baseline"
    else
        OUTPUT_DIR=".../checkpoints_codeonly/magicoder/${MODEL_NAME}/dp${TARGET_EPSILON}_baseline"
    fi

    # Run the finetune script using deepspeed
    deepspeed examples/codegen/finetune/finetune_dpsgd_baseline.py \
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

exit 0
