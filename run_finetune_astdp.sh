#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

# Script settings
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-15b"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
DATASET_NAME="ise-uiuc/Magicoder-OSS-Instruct-75K"

# Training settings
MAX_STEPS=0
BATCH_SIZE=8
GRAD_ACCUM_STEPS=16

LAMBDA=0.1
KL_STEP=40

# DP settings
TARGET_EPSILON=10
NON_PRIVATE="no"  # Set to "y" for non-private training

# Misc settings
LOG_FREQ=1
SAVE_FREQ=5
# SAVE_FREQ_EPOCH=1

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/${MODEL_NAME}/dp${TARGET_EPSILON}_lambda${LAMBDA}_klstep${KL_STEP}"

# Run the finetune script using deepspeed
deepspeed finetune_astdp.py \
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
    --lambda_kl $LAMBDA \
    --kl_step $KL_STEP \
    --deepspeed_config examples/codegen/finetune/config_stage2.json

exit 0
