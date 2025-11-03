#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Script settings
MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base" "Qwen/Qwen2.5-Coder-7B" "Qwen/CodeQwen1.5-7B" "google/codegemma-7b")

MODEL_PATH_STEP1="Qwen/Qwen2.5-Coder-1.5B"

# Training settings
MAX_STEPS=200
BATCH_SIZE=2
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILONs=( 1 )
NON_PRIVATE="y"

# ast settings
MAX_LAMBDAs=( 1000 )
ALPHAs=(0.01)

# round-trip settings
RT_MODEL="Llama-3.1-70B-Instruct"
# SIM_THRESHOLD=0.82
SIM_THRESHOLDs=(0.88)

# Misc settings
LOG_FREQ=1
SAVE_FREQ=50

for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
    for TARGET_EPSILON in "${TARGET_EPSILONs[@]}"; do
        for MODEL_PATH in "${MODEL_PATHS[@]}"; do
            for ALPHA in "${ALPHAs[@]}"; do
                for SIM_THRESHOLD in "${SIM_THRESHOLDs[@]}"; do
                    MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
                    MODEL_NAME_STEP1=$(echo $MODEL_PATH_STEP1 | awk -F '/' '{print $NF}')

                    DATASET_NAME="data/private_syn/${MODEL_NAME_STEP1}/${RT_MODEL}_tau${SIM_THRESHOLD}/final_dp${TARGET_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}.jsonl"
                    OUTPUT_DIR="checkpoints/privcode/utility_boosting/${MODEL_NAME}/${RT_MODEL}_tau${SIM_THRESHOLD}/dp${TARGET_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}"

                    # Run the finetune script using deepspeed
                    deepspeed examples/codegen/finetune/finetune_step2.py \
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
done

exit 0
