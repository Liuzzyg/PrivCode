#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
gpus=("0" "2" "3" "5" "4")
gpus=("2" "3" )
# gpus=("0" "1" "2" "3")
# gpus=("1" )

# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
MODEL_PATH="bigcode/starcoder2-3b"
MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilons=(10)


ALPHA=0.01  # the bigger, the more fastly lambda declines
MAX_LAMBDA=1000  # for ds-coder
# MAX_LAMBDA=1
MAX_LAMBDA=5  # for starcoder2-7b
MIN_LAMBDA=0.1


# steps=(40 50 45)
steps=(5 100 150)
# steps=(200 250 300 325)
steps=(30 40 50 60)
steps=(180 200)
steps=(110 90 130 160 170 190 180)
steps=(600 700 800 900 1000 1200 1300 750)

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
for dp_epsilon in "${dp_epsilons[@]}"; do
  for step in "${steps[@]}"; do
    for dataset in "${datasets[@]}"; do
      # declined lambda
      output_root="generate/bigcodebench/magicoder/${MODEL_NAME}/astdp/${dataset}"
      checkpoint_path=".../checkpoints_code/magicoder/${MODEL_NAME}/dp${dp_epsilon}_lambda${MAX_LAMBDA}to${MIN_LAMBDA}_alpha${ALPHA}_merged/checkpoint-${step}"
      

      # Define the command with parameters for evalplus.evaluate
      command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} bigcodebench.generate \
        --model \"${checkpoint_path}\" \
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
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
