#!/bin/bash

# Define GPU devices
gpus=("0" "1" "2" "3")

# Define multiple model paths for iteration
MODEL_PATHS=(
  "bigcode/starcoder2-3b" 
  "deepseek-ai/deepseek-coder-6.7b-base" 
  "Qwen/Qwen2.5-Coder-1.5B" 
  "deepseek-ai/deepseek-coder-1.3b-base"
)

# Static parameters
dataset="humaneval"
backend="vllm"
tp=1
greedy="--greedy"

# Initialize GPU index
gpu_index=0

# Run evaluation for each model path in MODEL_PATHS
for model_path in "${MODEL_PATHS[@]}"; do
  # Extract the model name from the path
  MODEL_NAME=$(echo $model_path | awk -F '/' '{print $NF}')
  
  # Define output root directory based on the model name
  output_root="generate/evalplus_0.3.1/${MODEL_NAME}/pretrain"

  # Define the command with parameters for evalplus.evaluate
  command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
    --model \"${model_path}\" \
    --root \"${output_root}\" \
    --dataset \"${dataset}\" \
    --backend \"${backend}\" \
    --tp ${tp} \
    ${greedy}"

  # Run command in the background
  echo "Running on GPU ${gpus[$gpu_index]}: $command"
  eval $command &

  # Update the GPU index to cycle through available GPUs
  gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))

  # Sleep to stagger execution slightly if needed
  sleep 1
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
