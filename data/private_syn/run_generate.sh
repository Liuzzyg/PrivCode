#!/bin/bash


MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

MAX_LAMBDAs=(1000)
ALPHAs=(0.01)
DP_EPSILONs=(4.0)
DATA_SIZEs=(55500)
# DP_EPSILON=1

STEP=100    # main
# STEP=1000   # epsilon hyper-parameter

NO_AST_FINETUNE_BASELINE='yes'
NO_AST_FINETUNE_BASELINE='no'

STABLE_LAMBDA='y'
# STABLE_LAMBDA='no'

HYPER_PARAMETER_ANALYSIS='y'
HYPER_PARAMETER_ANALYSIS='no'

for DP_EPSILON in "${DP_EPSILONs[@]}"; do
    for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
        for DATA_SIZE in "${DATA_SIZEs[@]}"; do
            for ALPHA in "${ALPHAs[@]}"; do
                if [[ "$NO_AST_FINETUNE_BASELINE" == "y" || "$NO_AST_FINETUNE_BASELINE" == "yes" ]]; then
                    CKPT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/magicoder/${MODEL_NAME}/dp${DP_EPSILON}_baseline_merged/checkpoint-${STEP}"
                    SAVE_PATH="data/private_syn/${MODEL_NAME}/codeonly/private_syndata_55k_dp${DP_EPSILON}_dpsgd_baseline_datasize${DATA_SIZE}.jsonl"
                elif [[ "$STABLE_LAMBDA" == "y" || "$STABLE_LAMBDA" == "yes" ]]; then
                    CKPT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/ablation/stable_lambda/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}_merged/checkpoint-${STEP}"
                    SAVE_PATH="data/private_syn/${MODEL_NAME}/codeonly/ablation/private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}_datasize${DATA_SIZE}.jsonl"
                elif [[ "$HYPER_PARAMETER_ANALYSIS" == "y" || "$HYPER_PARAMETER_ANALYSIS" == "yes" ]]; then
                    CKPT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/magicoder/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_merged/checkpoint-${STEP}"
                    SAVE_PATH="data/private_syn/${MODEL_NAME}/codeonly/hyper_parameter_analysis/private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                else
                    CKPT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/magicoder/${MODEL_NAME}/dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_merged/checkpoint-${STEP}"
                    SAVE_PATH="data/private_syn/${MODEL_NAME}/codeonly/private_syndata_55k_dp${DP_EPSILON}_lambda${MAX_LAMBDA}to0.1_alpha${ALPHA}_datasize${DATA_SIZE}.jsonl"
                fi

                CUDA_VISIBLE_DEVICES=0,1,2,3 python data/private_syn/generate.py \
                                                --ckpt "$CKPT" \
                                                --save_path "$SAVE_PATH" 
            done             
        done             
    done
done

exit 0