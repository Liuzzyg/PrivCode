#!/bin/bash

STEP1_MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
STEP1_MODEL_NAME=$(echo $STEP1_MODEL_PATH | awk -F '/' '{print $NF}')

RT_MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
# MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
RT_MODEL_NAME=$(echo $RT_MODEL_PATH | awk -F '/' '{print $NF}')

DP_EPSILONs=(0.2)
# DP_EPSILONs=(0.2 1.0 10.0)

MAX_LAMBDAs=( 100)
ALPHAs=(0.01)
DATA_SIZEs=(55500)
# DP_EPSILON=1

# SIM_THRESHOLDs=(0.82)   # main
SIM_THRESHOLDs=(0.5)   # ablation and hyper-parameter


# # main
GENERATED_NUM=1
TEMPERATURE=0


for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for DATA_SIZE in "${DATA_SIZEs[@]}"; do
            for SIM_THRESHOLD in "${SIM_THRESHOLDs[@]}"; do
                for ALPHA in "${ALPHAs[@]}"; do
                    INPUT_PATH="data/pii_dataset/privsyn/${STEP1_MODEL_NAME}/codeonly/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                    OUTPUT_PATH="data/pii_dataset/privsyn/${STEP1_MODEL_NAME}/codeonly/promptsim_${RT_MODEL_NAME}_tau${SIM_THRESHOLD}/final_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"

                    CUDA_VISIBLE_DEVICES=2,3 python data/pii_dataset/privsyn/round_trip_test_prompt.py \
                                --round_trip_model "$RT_MODEL_PATH" \
                                --input_path "$INPUT_PATH" \
                                --output_path "$OUTPUT_PATH" \
                                --sim_threshold "$SIM_THRESHOLD"\
                                --generated_num "$GENERATED_NUM"\
                                --temperature "$TEMPERATURE"
                done             
            done             
        done             
    done
done

exit 0

