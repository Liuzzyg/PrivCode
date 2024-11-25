#!/bin/bash

# Define parameters
gpus="0,1,2,3"
# gpus="4,5,6,7"
# gpus="0"

MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"


MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilons=(10)
# dp_epsilons=('inf')


# steps=(40 50 45)
# steps=(25 32 35 60 64 70 40)
# steps=(10 11 13 14 15 16 17 18 19 20 25)
steps=(110 90 130 160 170 190 180)
# steps=(30 40 60 70 80 100)
# steps=(12)
# steps=(30)
# steps=(50)

# Static parameters
datasets=("humaneval" "humanevalplus" "mbpp" "mbppplus")
datasets=("humaneval" "humanevalplus")
# datasets=("mbpp" "mbppplus")
# datasets=("humaneval")


max_workers=1


# Initialize GPU index and process counter
current_workers=0

# Run evaluation for each combination of dp_epsilon, lambda_kl, and step
for dataset in "${datasets[@]}"; do
  for dp_epsilon in "${dp_epsilons[@]}"; do
    for step in "${steps[@]}"; do
      # declined lambda
      checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/${MODEL_NAME}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
      output_root="generate/eval_harness/${MODEL_NAME}/dpbaseline/results_${dataset}_dp${dp_epsilon}_baseline_checkpoint-${step}.json"

      # Define the command with parameters for evalplus.evaluate
      command="CUDA_VISIBLE_DEVICES=${gpus} accelerate launch  bigcode-evaluation-harness/main.py \
        --model \"${checkpoint_path}\" \
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
  done
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
