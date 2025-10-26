#!/bin/bash

MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base" "Qwen/Qwen2.5-Coder-7B"  "Qwen/CodeQwen1.5-7B" "google/codegemma-7b")

MODEL_PATHS=("google/codegemma-7b" "Qwen/CodeQwen1.5-7B")
# MODEL_PATHS=("Qwen/CodeQwen1.5-7B")
MODEL_PATHS=("Qwen/Qwen2.5-Coder-7B" )

# MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base")

# GPU configuration
gpus=("2" "3")
gpus=("0" "1" "2" "3")
# gpus=("2" "1")

MAX_LAMBDAs=(100)
ALPHAs=(0.01)
DATA_SIZEs=(55500)

DP_EPSILONs=(0.2 10)
DP_EPSILONs=( 10)
DP_EPSILONs=( 4)
# DP_EPSILONs=( 1 4 10)
# DP_EPSILONs=(0.2 1 4)

REPs=(1 10 100)
REPs=(100)
TEMPERATUREs=(0.2 0.6)
TEMPERATUREs=(1)
TOP_Ps=(0.5 0.8)
TOP_Ps=(1)

STEPs=( 100 150 200)
STEPs=( 110)

SEEDS=(22)
SEEDS=(1 5)

max_workers=4


gpu_index=0
current_workers=0

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

    for SEED in "${SEEDS[@]}"; do
        for STEP in "${STEPs[@]}"; do
            for DP_EPSILON in "${DP_EPSILONs[@]}"; do
                for MAX_LAMBDA in "${MAX_LAMBDAs[@]}"; do
                    for DATA_SIZE in "${DATA_SIZEs[@]}"; do
                        for ALPHA in "${ALPHAs[@]}"; do
                            for REP in "${REPs[@]}"; do
                                for TEMPERATURE in "${TEMPERATUREs[@]}"; do
                                    for TOP_P in "${TOP_Ps[@]}"; do

                                        CKPT=".../dpcode/checkpoints_codeonly/canary/${MODEL_NAME}/dpinf_baseline_rep${REP}_merged/checkpoint-${STEP}"

                                        SAVE_PATH="canary/eval/detect_results_python/step2_infbaseline/${MODEL_NAME}/seed${SEED}/dpinf_baseline_rep${REP}_step${STEP}_temp${TEMPERATURE}_topp${TOP_P}.jsonl"

                                        if [ "$TEMPERATURE" == "0" ]; then
                                            # Assign a GPU and execute the command
                                            command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python canary/eval/detector.py \
                                                --model_path \"$CKPT\" \
                                                --output_path \"$SAVE_PATH\" \
                                                --temperature "$TEMPERATURE" \
                                                --top_p "$TOP_P" \
                                                --n 1 \
                                                --seed "$SEED" \
                                                --prompt_num 128"
                                        else
                                            command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} python canary/eval/detector.py \
                                                --model_path \"$CKPT\" \
                                                --output_path \"$SAVE_PATH\" \
                                                --temperature "$TEMPERATURE" \
                                                --top_p "$TOP_P" \
                                                --seed "$SEED" \
                                                --prompt_num 128"
                                        fi

                                        echo "Running on GPU ${gpus[$gpu_index]}: $command"
                                        eval $command &

                                        # Increment the worker count
                                        gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
                                        current_workers=$((current_workers + 1))

                                        # If current workers reach max_workers, wait for some to finish
                                        if (( current_workers >= max_workers )); then
                                            wait -n  # Wait for any background process to complete
                                            current_workers=$((current_workers - 1))  # Decrease worker count after a process finishes
                                        fi

                                        # Optional: slight delay between task starts
                                        sleep 1
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Final wait for all processes to complete
wait
echo "All tasks have completed."