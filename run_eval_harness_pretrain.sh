#!/bin/bash

# Define parameters
gpus="0,1,2,3"
MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"


MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

# Static parameters
datasets=("humaneval" "humanevalplus" "mbpp" "mbppplus")
# datasets=("mbpp" "mbppplus")
datasets=( "humanevalplus" "mbpp" "mbppplus")


max_workers=1


# Initialize GPU index and process counter
current_workers=0

# Run evaluation for each combination of dp_epsilon, lambda_kl, and step
for dataset in "${datasets[@]}"; do
  # declined lambda
  output_root="generate/eval_harness/${MODEL_NAME}/pretrain/results_${dataset}.json"

  # Define the command with parameters for evalplus.evaluate
  command="CUDA_VISIBLE_DEVICES=${gpus} accelerate launch  bigcode-evaluation-harness/main.py \
    --model \"${MODEL_PATH}\" \
    --max_length_generation 512   \
    --tasks \"${dataset}\"   \
    --n_samples 1   \
    --batch_size 1   \
    --temperature 0   \
    --do_sample False   \
    --precision bf16   \
    --allow_code_execution   \
    --use_auth_token       \
    --metric_output_path \"${output_root}\""


  # Run command in the background
  eval $command &

  # Increment the GPU index and worker count
  current_workers=$((current_workers + 1))

  # If current workers reach max_workers, wait for some to finish
  if (( current_workers >= max_workers )); then
    wait -n  # Wait for any background process to complete
    current_workers=$((current_workers - 1))  # Decrease worker count after a process finishes
  fi

  # Optional sleep to stagger execution slightly if needed
  sleep 1
done
# Wait for all background processes to complete
wait

echo "All tasks have completed."
