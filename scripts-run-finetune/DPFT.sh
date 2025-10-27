#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Script settings
MODEL_PATHS=("Qwen/Qwen2.5-Coder-7B")
MODEL_PATH_STEP1="Qwen/Qwen2.5-Coder-1.5B"

# Training settings
MAX_GLOBAL_STEPS=2000
BATCH_SIZE=4
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILONs=(4)
NON_PRIVATE="no"

# ast settings
MAX_LAMBDAs=(1000)
ALPHAs=(0.01)
DATA_SIZEs=(55500)

# round-trip settings
RT_MODEL="Llama-3.1-70B-Instruct"
SIM_THRESHOLD=0.82

# Misc settings
LOG_FREQ=1
SAVE_FREQ=100

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for TARGET_EPSILON in "${TARGET_EPSILONs[@]}"; do
        for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
            for ALPHA in "${ALPHAs[@]}"; do
                for DATA_SIZE in "${DATA_SIZEs[@]}"; do
                    MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
                    MODEL_NAME_STEP1=$(echo $MODEL_PATH_STEP1 | awk -F '/' '{print $NF}')

                    DATASET_NAME="ise-uiuc/Magicoder-OSS-Instruct-75K"
                    OUTPUT_DIR="checkpoints/${MODEL_NAME}/dp${TARGET_EPSILON}_baseline"

                    # Run the finetune script using deepspeed
                    deepspeed examples/codegen/finetune/finetune_step2.py \
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
                        --deepspeed_config examples/codegen/finetune/config_stage2.json
                done
            done
        done
    done
done

exit 0
