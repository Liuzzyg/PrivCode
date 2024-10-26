#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

# Script settings
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# DATASET_NAME="pii_leaks_eval/pii_dataset/pii_dataset.jsonl"
DATASET_NAME="terryyz/pii"

SEQ_LEN=2048

# Training settings
MAX_STEPS=0
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16

LAMBDA=0.02

# DP settings
TARGET_EPSILON=10
# NON_PRIVATE="no"  # Set to "y" for non-private training
NON_PRIVATE="no"

# Misc settings
LOG_FREQ=1
SAVE_FREQ=5
# SAVE_FREQ_EPOCH=1

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/pii_data/${MODEL_NAME}/dp${TARGET_EPSILON}_lambda${LAMBDA}"
# OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/pii_data/${MODEL_NAME}/dpinf_lambda${LAMBDA}"

# Run the finetune script using deepspeed
deepspeed finetune_astdp.py \
    --model_path $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --seq_length $SEQ_LEN \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --output_dir $OUTPUT_DIR \
    --log_freq $LOG_FREQ \
    --save_freq $SAVE_FREQ \
    --target_epsilon $TARGET_EPSILON \
    --non_private $NON_PRIVATE \
    --lambda_kl $LAMBDA \
    --deepspeed_config examples/codegen/finetune/config_stage2.json

exit 0
