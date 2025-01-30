#!/bin/bash

STEP1_MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
STEP1_MODEL_NAME=$(echo $STEP1_MODEL_PATH | awk -F '/' '{print $NF}')

RT_MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct"
# MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
RT_MODEL_NAME=$(echo $RT_MODEL_PATH | awk -F '/' '{print $NF}')

DP_EPSILONs=(4.0)
# DP_EPSILONs=(0.2 1.0 10.0)

MAX_LAMBDAs=( 1000)
ALPHAs=(0.01)
DATA_SIZEs=(55500)
# DP_EPSILON=1

# SIM_THRESHOLDs=(0.82)   # main
SIM_THRESHOLDs=(0.88)   # ablation and hyper-parameter

# NO_AST_FINETUNE_BASELINE='y'
NO_AST_FINETUNE_BASELINE='no'

STABLE_LAMBDA='y'
# STABLE_LAMBDA='no'

# HYPER_PARAMETER_ANALYSIS='y'
HYPER_PARAMETER_ANALYSIS='no'


# # main
# GENERATED_NUM=20
# TEMPERATURE=0.8

# ablation and hyper-parameter
GENERATED_NUM=1
TEMPERATURE=0

for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for DATA_SIZE in "${DATA_SIZEs[@]}"; do
            for SIM_THRESHOLD in "${SIM_THRESHOLDs[@]}"; do
                for ALPHA in "${ALPHAs[@]}"; do
                    if [[ "$NO_AST_FINETUNE_BASELINE" == "y" || "$NO_AST_FINETUNE_BASELINE" == "yes" ]]; then
                        INPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/ablation/cleaned_private_syndata_55k_dp${DP_EPSILON}_dpsgd_baseline_datasize${DATA_SIZE}.jsonl"
                        OUTPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/ablation/promptsim_${RT_MODEL_NAME}_tau${SIM_THRESHOLD}/final_private_syndata_55k_dp${DP_EPSILON}_dpsgd_baseline_datasize${DATA_SIZE}.jsonl"
                    elif [[ "$STABLE_LAMBDA" == "y" || "$STABLE_LAMBDA" == "yes" ]]; then
                        INPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/ablation/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}_datasize${DATA_SIZE}.jsonl"
                        OUTPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/ablation/promptsim_${RT_MODEL_NAME}_tau${SIM_THRESHOLD}/final_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}_datasize${DATA_SIZE}.jsonl"
                    elif [[ "$HYPER_PARAMETER_ANALYSIS" == "y" || "$HYPER_PARAMETER_ANALYSIS" == "yes" ]]; then
                        INPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/hyper_parameter_analysis/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                        OUTPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/hyper_parameter_analysis/promptsim_${RT_MODEL_NAME}_tau${SIM_THRESHOLD}/final_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                    else
                        INPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/cleaned_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                        OUTPUT_PATH="data/private_syn/${STEP1_MODEL_NAME}/codeonly/promptsim_${RT_MODEL_NAME}_tau${SIM_THRESHOLD}/final_private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                    fi

                    CUDA_VISIBLE_DEVICES=0,1,2,3 python data/private_syn/round_trip_test_prompt.py \
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

