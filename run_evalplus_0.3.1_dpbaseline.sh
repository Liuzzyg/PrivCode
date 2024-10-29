#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
# gpus=("0" "1" "2" "3")
gpus=("0" "1" "2" "3" "4" "5" "6" "7")

# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilons=(10)
# dp_epsilons=('inf')
steps=(505 510 515 520 495 490 485 480)
steps=(765 755 770 790 810 840 835 830)
# steps = [210]

# Static parameters
output_root="generate/evalplus_0.3.1/${MODEL_NAME}/dpbaseline"
dataset="humaneval"
backend="vllm"
tp=1
greedy="--greedy"

max_workers=8

# Initialize GPU index
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, and step
for dp_epsilon in "${dp_epsilons[@]}"; do
  for step in "${steps[@]}"; do
    # Set the output path and checkpoint path based on current parameters
    # dp baseline
    checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/${MODEL_NAME}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
    
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

# Wait for all background processes to complete
wait

echo "All tasks have completed."
