#!/bin/bash


MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

MAX_LAMBDAs=(100)
ALPHAs=(0.01)
DATA_SIZEs=(55500)

DP_EPSILONs=(0.2 1 4 10)
# DP_EPSILONs=(0.2)

STEPs=(500 600)    # main


for STEP in "${STEPs[@]}"; do
    for DP_EPSILON in "${DP_EPSILONs[@]}"; do
        for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
            for DATA_SIZE in "${DATA_SIZEs[@]}"; do
                for ALPHA in "${ALPHAs[@]}"; do
                    
                    CKPT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/pii_dataset/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_merged/checkpoint-${STEP}"
                    if [[ "$MODEL_NAME" == "Qwen2.5-Coder-1.5B" ]]; then
                        SAVE_PATH="pii_leaks_eval/detect_results/step1/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}_step${STEP}.jsonl"
                    else
                        SAVE_PATH="pii_leaks_eval/detect_results/step2/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}_step${STEP}.jsonl"
                    fi

                    CUDA_VISIBLE_DEVICES=1 python pii_leaks_eval/detector.py \
                                                    --model_path "$CKPT" \
                                                    --output_path "$SAVE_PATH" \
                                                    --prompt_num 128
                done             
            done             
        done
    done
done


exit 0


