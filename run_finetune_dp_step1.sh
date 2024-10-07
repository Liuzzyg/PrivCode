#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2

# Script settings
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct
DATASET_NAME="ise-uiuc/Magicoder-OSS-Instruct-75K"

# Training settings
MAX_STEPS=50
BATCH_SIZE=2
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILON=10
NON_PRIVATE="no"  # Set to "y" for non-private training

# Misc settings
LOG_FREQ=1
SAVE_FREQ=5
# SAVE_FREQ_EPOCH=1

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
OUTPUT_DIR="checkpoints_step1/magicoder/${MODEL_NAME}/dp${TARGET_EPSILON}"

# Run the finetune script using deepspeed
deepspeed finetune_dp_step1.py \
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
    --deepspeed_config examples/codegen/finetune/config_stage1.json

exit 0