#!/bin/bash

MODEL_NAME="Qwen2.5-Coder-1.5B"

DP_EPSILONs=(4)

MAX_LAMBDAs=(1000)
ALPHAs=(0.01)

for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for ALPHA in "${ALPHAs[@]}"; do
            INPUT_PATH="data/private_syn/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}.jsonl"
            OUTPUT_PATH="data/private_syn/${MODEL_NAME}/cleaned_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}.jsonl"

            python data/private_syn/exe_clean_data.py \
                                --input_file "$INPUT_PATH" \
                                --output_file "$OUTPUT_PATH" &
        done         
    done
done

wait

exit 0