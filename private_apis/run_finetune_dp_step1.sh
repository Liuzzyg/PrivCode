#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Script settings
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
DATASET_NAME="data/oss_instruction/valid_processed_instruction_data_25k.jsonl"

# Training settings
MAX_STEPS=0
BATCH_SIZE=8
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILON=10
NON_PRIVATE="no"  # Set to "y" for non-private training
LOGIC_BS=32

# Misc settings
LOG_FREQ=1
SAVE_FREQ=5
# SAVE_FREQ_EPOCH=1

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/private_api_numpy/${MODEL_NAME}/dp${TARGET_EPSILON}_lbs${LOGIC_BS}"

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
    --deepspeed_config examples/codegen/finetune/config_stage1.json \
    --logical_batch_size $LOGIC_BS \

exit 0
