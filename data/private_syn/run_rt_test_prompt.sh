#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

STEP1_MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
STEP1_MODEL_NAME=$(echo $STEP1_MODEL_PATH | awk -F '/' '{print $NF}')

RT_MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
RT_MODEL_NAME=$(echo $RT_MODEL_PATH | awk -F '/' '{print $NF}')

DP_EPSILONs=(4)

MAX_LAMBDAs=(1000)
ALPHAs=(0.01)
DATA_SIZEs=(55500)

# round trip
SIM_THRESHOLDs=(0.88)
GENERATED_NUM=20
TEMPERATURE=0.8


for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for DATA_SIZE in "${DATA_SIZEs[@]}"; do
            for SIM_THRESHOLD in "${SIM_THRESHOLDs[@]}"; do
                for ALPHA in "${ALPHAs[@]}"; do
                    INPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/cleaned_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}.jsonl"
                    OUTPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/${RT_MODEL_NAME}_tau${SIM_THRESHOLD}/final_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}.jsonl"

                    python data/private_syn/round_trip_test_prompt.py \
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

