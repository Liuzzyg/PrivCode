#!/bin/bash

# Define GPU devices
gpus=("4" "5" "6" "7")
gpus=("0" "1")

# Define multiple model paths for iteration
MODEL_PATHS=(
  # "bigcode/starcoder2-3b" 
  # "bigcode/starcoder2-7b" 
  # "deepseek-ai/deepseek-coder-1.3b-base"
  # "deepseek-ai/deepseek-coder-6.7b-base" 
  # "Qwen/Qwen2.5-Coder-1.5B" 
  # "Qwen/Qwen2.5-Coder-7B" 
  # "google/codegemma-7b"
  "Qwen/CodeQwen1.5-7B"
)

# Static parameters
datasets=("humaneval" "mbpp")
datasets=("mbpp")
datasets=("humaneval")
backend="vllm"
tp=1
greedy="--greedy"

max_workers=4


# Initialize GPU index
gpu_index=0
current_workers=0

# Run evaluation for each model path in MODEL_PATHS
for model_path in "${MODEL_PATHS[@]}"; do
  for dataset in "${datasets[@]}"; do
    # Extract the model name from the path
    MODEL_NAME=$(echo $model_path | awk -F '/' '{print $NF}')
    
    # Define output root directory based on the model name
    output_root="generate/evalplus_0.3.1/${MODEL_NAME}/pretrain_chat"

    # Define the command with parameters for evalplus.evaluate
    command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
      --model \"${model_path}\" \
      --root \"${output_root}\" \
      --dataset \"${dataset}\" \
      --backend \"${backend}\" \
      --tp ${tp} \
      ${greedy}"

    # command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
    #   --model \"${model_path}\" \
    #   --root \"${output_root}\" \
    #   --dataset \"${dataset}\" \
    #   --backend \"${backend}\" \
    #   --tp ${tp} \
    #   --force-base-prompt \
    #   ${greedy}"

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
    
    # Sleep to stagger execution slightly if needed
    sleep 1
  done
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
