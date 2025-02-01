#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
gpus=("0" "2" "3" "5" "4")
gpus=("1")
# gpus=("0" "1" "2" "3")
# gpus=("1" )

MODEL_PATHS=(
  # "deepseek-ai/deepseek-coder-6.7b-base"
  "deepseek-ai/deepseek-coder-1.3b-base"
  # "bigcode/starcoder2-3b"
  # "bigcode/starcoder2-7b"
  # "Qwen/Qwen2.5-Coder-1.5B"
  # "Qwen/Qwen2.5-Coder-7B"
)


# Static parameters
datasets=("full" "hard")
datasets=("full")
# datasets=("hard")

split='complete'

backend="vllm"
# backend="hf"
tp=1
BATCH_SIZE=2
greedy="--greedy"
resume="--resume"


max_workers=2


# Initialize GPU index and process counter
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, lambda_kl, and step
for dataset in "${datasets[@]}"; do
  for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
    output_root="generate/bigcodebench/magicoder/${MODEL_NAME}/pretrain/${dataset}"

    # Define the command with parameters for evalplus.evaluate
    command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} bigcodebench.generate \
      --model \"${MODEL_PATH}\" \
      --model_name \"${MODEL_NAME}\" \
      --split \"${split}\" \
      --subset \"${datasets}\" \
      --bs \"${BATCH_SIZE}\" \
      --backend \"${backend}\" \
      --save_root_path \"${output_root}\" \
      --step \"${step}\" \
      ${greedy} ${resume}" 


    # Run command in the background
    echo "Running on GPU ${gpus[$gpu_index]}: $command"
    eval $command &

    # Increment the GPU index and worker count
    gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
    current_workers=$((current_workers + 1))

    # If current workers reach max_workers, wait for some to finish
    if (( current_workers >= max_workers )); then
      wait -n  # Wait for any background process to complete
      current_workers=$((current_workers - 1))  # Decrease worker count after a process finishes
    fi

    # Optional sleep to stagger execution slightly if needed
    sleep 1
  done
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."