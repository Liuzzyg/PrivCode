#!/bin/bash


MODEL_NAME="Qwen2.5-Coder-1.5B"

DP_EPSILONs=(4.0)
DP_EPSILONs=(1 4 10)

MAX_LAMBDAs=(100)
# MAX_LAMBDAs=(1 100000)  # ablation
ALPHAs=(0.01)
DATA_SIZEs=(55500)

# NO_AST_FINETUNE_BASELINE='y'
NO_AST_FINETUNE_BASELINE='no'

STABLE_LAMBDA='y'
STABLE_LAMBDA='no'

# HYPER_PARAMETER_ANALYSIS='y'
HYPER_PARAMETER_ANALYSIS='no'

for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for DATA_SIZE in "${DATA_SIZEs[@]}"; do
            for ALPHA in "${ALPHAs[@]}"; do
                if [[ "$NO_AST_FINETUNE_BASELINE" == "y" || "$NO_AST_FINETUNE_BASELINE" == "yes" ]]; then
                    INPUT_PATH="data/private_syn/${MODEL_NAME}/codeonly/ablation/private_syndata_55k_dp${DP_EPSILON}_dpsgd_baseline_datasize${DATA_SIZE}.jsonl"
                    OUTPUT_PATH="data/private_syn/${MODEL_NAME}/codeonly/ablation/cleaned_private_syndata_55k_dp${DP_EPSILON}_dpsgd_baseline_datasize${DATA_SIZE}.jsonl"
                elif [[ "$STABLE_LAMBDA" == "y" || "$STABLE_LAMBDA" == "yes" ]]; then
                    INPUT_PATH="data/private_syn/${MODEL_NAME}/codeonly/ablation/private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}_datasize${DATA_SIZE}.jsonl"
                    OUTPUT_PATH="data/private_syn/${MODEL_NAME}/codeonly/ablation/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}_datasize${DATA_SIZE}.jsonl"
                elif [[ "$HYPER_PARAMETER_ANALYSIS" == "y" || "$HYPER_PARAMETER_ANALYSIS" == "yes" ]]; then
                    INPUT_PATH="data/private_syn/${MODEL_NAME}/codeonly/hyper_parameter_analysis/private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                    OUTPUT_PATH="data/private_syn/${MODEL_NAME}/codeonly/hyper_parameter_analysis/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                else
                    INPUT_PATH="data/canary/${MODEL_NAME}/codeonly/private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                    OUTPUT_PATH="data/canary/${MODEL_NAME}/codeonly/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                fi

                python data/pii_dataset/privsyn/exe_clean_data.py \
                                    --input_file "$INPUT_PATH" \
                                    --output_file "$OUTPUT_PATH" 
            done             
        done             
    done
done

wait

exit 0