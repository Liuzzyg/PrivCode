#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
# gpus=("0" "1" "2" "3" "4" "5" "6")
gpus=("0" "1" "2" "3")

# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilons=(10)
lambda_kl=(0.1)
kl_steps=(20 30 40)
# steps=(200 250 300 350 400 450 485)
steps=(90 100 110 120)

max_workers=4

# Static parameters
output_root="generate/evalplus_0.3.1/${MODEL_NAME}/astdp"
dataset="humaneval"
backend="vllm"
tp=1
greedy="--greedy"

# Initialize GPU index and process counter
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, lambda_kl, and step
for dp_epsilon in "${dp_epsilons[@]}"; do
  for lam in "${lambda_kl[@]}"; do
    for step in "${steps[@]}"; do
      for kl_step in "${kl_steps[@]}"; do
        # Set the output path and checkpoint path based on current parameters
        checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/${MODEL_NAME}/dp${dp_epsilon}_lambda${lam}_klstep${kl_step}_merged/checkpoint-${step}"

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

        # Optional sleep to stagger execution slightly if needed
        sleep 1
      done
    done
  done
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
