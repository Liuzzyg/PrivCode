#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0

# Script settings
DATASET_NAME="ise-uiuc/Magicoder-OSS-Instruct-75K"

# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
MODEL_PATH="Qwen/CodeQwen1.5-7B"

MODEL_PATHS=("Qwen/CodeQwen1.5-7B")
# MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base" "Qwen/Qwen2.5-Coder-7B" "Qwen/CodeQwen1.5-7B" "google/codegemma-7b")
# MODEL_PATHS=("Qwen/CodeQwen1.5-7B" "google/codegemma-7b" )
# MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base" )
# MODEL_PATHS=("Qwen/Qwen2.5-Coder-7B" )

MODEL_PATH_STEP1="Qwen/Qwen2.5-Coder-1.5B"

# Training settings
MAX_STEPS=2500
BATCH_SIZE=2
GRAD_ACCUM_STEPS=16

# DP settings
TARGET_EPSILONs=(1 4 10)
TARGET_EPSILONs=(4)
NON_PRIVATE="no"

# ast settings
MAX_LAMBDAs=(100 )
ALPHAs=(0.01)
DATA_SIZEs=(55500)

# round-trip settings
RT_MODEL="Llama-3.1-70B-Instruct"
SIM_THRESHOLD=0.5

# Misc settings
LOG_FREQ=1
SAVE_FREQ=10
# SAVE_FREQ_EPOCH=1

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for TARGET_EPSILON in "${TARGET_EPSILONs[@]}"; do
        for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
            for ALPHA in "${ALPHAs[@]}"; do
                for DATA_SIZE in "${DATA_SIZEs[@]}"; do
                    MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
                    MODEL_NAME_STEP1=$(echo $MODEL_PATH_STEP1 | awk -F '/' '{print $NF}')

                    CHECKPOINT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/baseline/jft/${MODEL_NAME}/dpinf_step1_merged/checkpoint-150"

                    OUTPUT_DIR="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/baseline/jft/${MODEL_NAME}/dp4_step2"

                    # Run the finetune script using deepspeed
                    deepspeed finetune_step2_load_checkpoint.py \
                        --model_path $MODEL_PATH \
                        --dataset_name $DATASET_NAME \
                        --resume_from_checkpoint $CHECKPOINT \
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
