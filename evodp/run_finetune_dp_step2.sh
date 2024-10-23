#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2

# Script settings
MODEL_PATH_STEP2="deepseek-ai/deepseek-coder-6.7b-base"
MODEL_PATH_STEP1="deepseek-ai/deepseek-coder-1.3b-instruct"

# Training settings
MAX_STEPS=0   # 0 for epoch
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILON=10
NON_PRIVATE="y"  # Set to "y" for non-private training

# Misc settings
LOG_FREQ=1
SAVE_FREQ=10
# SAVE_FREQ_EPOCH=1

MODEL_NAME_STEP2=$(echo $MODEL_PATH_STEP2 | awk -F '/' '{print $NF}')
MODEL_NAME_STEP1=$(echo $MODEL_PATH_STEP1 | awk -F '/' '{print $NF}')

DATASET_NAME="data/private_syn/${MODEL_NAME_STEP1}_cleaned_private_syndata.jsonl"
OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/${MODEL_NAME_STEP2}/dp${TARGET_EPSILON}_syndata"


# DATASET_NAME="data/private_syn/${MODEL_NAME_STEP1}_original_data.jsonl"
# OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/${MODEL_NAME_STEP2}/original_data"

# Run the finetune script using deepspeed
deepspeed finetune_dp_step2.py \
    --model_path $MODEL_PATH_STEP2 \
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
