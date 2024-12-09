#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
gpus=("2" "3")
# gpus=("2" "3" "4" "5" "6" "7")
gpus=("2")
# gpus=("0" "1" "2" )

MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
global_dp_epsilon=4
dp_epsilons=(4)
dp_epsilons=('inf')

# steps=(50 40 30 20 100 10)
# steps=(950 800 600 400)
# steps=(120 100 80)
# steps=(50 60 70)
# steps=(1100 1300 1500 1700 1200 1400 1600 1000)
# steps=(600 700 900  1200  750)
# steps=(110 90 130 160 170 190 180)
steps=(100 200 500 750 1000 1500 2000)
steps=(2000)
# steps=(5 10 15 20 25 30 40 45 50)
# steps=(5 10 15 20 25 30 35 40)
steps=(500)

# Static parameters
output_root="generate/evalplus_0.3.1/${MODEL_NAME}/step2"
datasets=("humaneval" "mbpp")
# datasets=("mbpp")
# datasets=("humaneval")

is_baseline='yes'
is_baseline='no'

backend="vllm"
# backend="hf"
tp=1
greedy="--greedy"

max_workers=1

# Initialize GPU index
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, and step
for dp_epsilon in "${dp_epsilons[@]}"; do
  for step in "${steps[@]}"; do
    for dataset in "${datasets[@]}"; do
      # Set the output path and checkpoint path based on current parameters
      # dp baseline
      if [ "$dp_epsilon" == "inf" ]; then
          if [ "$is_baseline" == "no" ]; then
              checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2_final_filtered/${MODEL_NAME}_dp${global_dp_epsilon}/privsyn_merged/checkpoint-${step}"
          else 
              checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2_final_filtered/${MODEL_NAME}_dp${global_dp_epsilon}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
          fi
      else
          checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2_final_filtered/${MODEL_NAME}_dp${global_dp_epsilon}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
      fi

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
