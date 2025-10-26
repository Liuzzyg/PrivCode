#!/bin/bash

# Define parameters
gpus=("6" "7")
# gpus=("0" "1" "2" "3")
gpus=("1" "2" "3" "4" "5" "6" "7")
# gpus=("0" "1")
# gpus=("0" "1" "2" )

# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
MODEL_PATH="bigcode/starcoder2-3b"
MODEL_PATH="bigcode/starcoder2-7b"
MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilons=(1)
dp_epsilons=('inf')

# steps=(50 40 30 20 100 10)
# steps=(40 50 80 100)
steps=(950 800 600 400)
steps=(120 100 80)
steps=(50 60 70)
steps=(1100 1300 1500 1700 1200 1400 1600 1000)
steps=(100)
# steps=(15)

# Static parameters
output_root="generate/evalplus_0.3.1/${MODEL_NAME}/dpbaseline"
datasets=("humaneval" "mbpp")
# datasets=("mbpp")
datasets=("humaneval")

backend="vllm"
# backend="hf"
tp=1
greedy="--greedy"

max_workers=10

# Initialize GPU index
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, and step
for dp_epsilon in "${dp_epsilons[@]}"; do
  for step in "${steps[@]}"; do
    for dataset in "${datasets[@]}"; do
      # Set the output path and checkpoint path based on current parameters
      # dp baseline
      checkpoint_path=".../checkpoints_codeonly/magicoder/${MODEL_NAME}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"

      # Define the command with parameters for evalplus.evaluate
      command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
        --model \"${checkpoint_path}\" \
        --root \"${output_root}\" \
        --dataset \"${dataset}\" \
        --backend \"${backend}\" \
        --tp ${tp} \
        ${greedy}"

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
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
