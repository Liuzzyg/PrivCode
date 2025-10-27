#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

MAX_LAMBDAs=(1000)
ALPHAs=(0.01)
DP_EPSILONs=(4)
STEP=100

for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for ALPHA in "${ALPHAs[@]}"; do
            CKPT="checkpoints/privcode/privacy_sanitizing/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_merged/checkpoint-${STEP}"
            SAVE_PATH="data/private_syn/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}.jsonl"

            python data/private_syn/generate.py \
                --ckpt "$CKPT" \
                --save_path "$SAVE_PATH" 
        done            
    done
done

exit 0