#!/bin/bash

# Define parameters
gpus=("4" "5" "6" "7")
# gpus=("0" "2" "3" "5" "4")
gpus=( "1" )
# gpus=("0" "1" "2" "3" "4" "5" "6" "7")
gpus=("0" "2")


MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilons=(0.2 1 4 10)
dp_epsilons=(0.2  4 )

# lambda_kl=(0.1)
lambda_kl=(1000)
# kl_steps=(20 30 40)
kl_steps=(10000)
# kl_steps=(2)

ALPHA=0.01  # the bigger, the more fastly lambda declines
MAX_LAMBDA=100  # for ds-coder
# MAX_LAMBDA=1
# MAX_LAMBDA=5  # for starcoder2-7b
MIN_LAMBDA=0.1


steps=(20 30 40 )

# Static parameters
output_root="generate/evalplus_0.3.1/pii_python/${MODEL_NAME}/astdp"
datasets=("humaneval" "mbpp")
# datasets=("mbpp")
datasets=("humaneval")
backend="vllm"
# backend="hf"
tp=1
greedy="--greedy"


max_workers=2


# Initialize GPU index and process counter
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, lambda_kl, and step
for dp_epsilon in "${dp_epsilons[@]}"; do
  for lam in "${lambda_kl[@]}"; do
    for step in "${steps[@]}"; do
      for kl_step in "${kl_steps[@]}"; do
        for dataset in "${datasets[@]}"; do
          # declined lambda
          checkpoint_path=".../checkpoints_codeonly/pii_dataset_python/${MODEL_NAME}/dp${dp_epsilon}_lambda${MAX_LAMBDA}to${MIN_LAMBDA}_alpha${ALPHA}_merged/checkpoint-${step}"
          

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
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
