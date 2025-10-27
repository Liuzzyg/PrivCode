#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1

# Script settings
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
DATASET_NAME="ise-uiuc/Magicoder-OSS-Instruct-75K"

# Training settings
MAX_GLOBAL_STEPS=100
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16

# the bigger, the more fastly lambda declines
ALPHA=0.01
MAX_LAMBDAs=(1000)
MIN_LAMBDA=0.1

# DP settings
TARGET_EPSILONs=(4)
NON_PRIVATE="no"  # Set to "y" for non-private training

# Misc settings
LOG_FREQ=1
SAVE_FREQ=1

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
    for TARGET_EPSILON in "${TARGET_EPSILONs[@]}"; do
        OUTPUT_DIR="checkpoints/privcode/privacy_sanitizing/${MODEL_NAME}/dp${TARGET_EPSILON}_lambda${MAX_LAMBDA}to${MIN_LAMBDA}_alpha${ALPHA}"

        # Run the finetune script using deepspeed
        deepspeed examples/codegen/finetune/finetune_astdp.py \
            --model_path $MODEL_PATH \
            --dataset_name $DATASET_NAME \
            --max_steps $MAX_GLOBAL_STEPS \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
            --output_dir $OUTPUT_DIR \
            --log_freq $LOG_FREQ \
            --save_freq $SAVE_FREQ \
            --target_epsilon $TARGET_EPSILON \
            --non_private $NON_PRIVATE \
            --alpha $ALPHA \
            --min_lambda_kl $MIN_LAMBDA \
            --max_lambda_kl $MAX_LAMBDA \
            --deepspeed_config examples/codegen/finetune/config_stage2.json
    done
done

exit 0
