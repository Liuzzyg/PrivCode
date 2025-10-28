#!/bin/bash

gpus=("0" "1")

MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
dp_epsilon=4

steps=(100)

# round-trip configs
round_trip_model="Llama-3.1-70B-Instruct"
sim_thresholds=(0.88)

# ast configs
alphas=(0.01)
max_lambdas=(1000)

# Static parameters
datasets=("humaneval" "mbpp")

# set chat_model to 'y' means instruction following task, 'n' means code completion task
chat_model='n'

backend="vllm"
tp=1
greedy="--greedy"

max_workers=2

gpu_index=0
current_workers=0

for sim_threshold in "${sim_thresholds[@]}"; do
  for alpha in "${alphas[@]}"; do
    for max_lambda in "${max_lambdas[@]}"; do
      for step in "${steps[@]}"; do
        for dataset in "${datasets[@]}"; do
          checkpoint_path="checkpoints/privcode/utility_boosting/${MODEL_NAME}/${round_trip_model}_tau${sim_threshold}/dp${dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_merged/checkpoint-${step}"

          if [ "$chat_model" == "y" ]; then
            output_root="results/privcode/${MODEL_NAME}/evalplus/instruct"
            command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
              --model \"${checkpoint_path}\" \
              --root \"${output_root}\" \
              --dataset \"${dataset}\" \
              --backend \"${backend}\" \
              --tp ${tp} \
              ${greedy}"
          else
            output_root="results/privcode/${MODEL_NAME}/evalplus/complete"
            command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
              --model \"${checkpoint_path}\" \
              --root \"${output_root}\" \
              --dataset \"${dataset}\" \
              --backend \"${backend}\" \
              --force-base-prompt \
              --tp ${tp} \
              ${greedy}"
          fi

          echo "Running on GPU ${gpus[$gpu_index]}: $command"
          eval $command &

          gpu_index=$(( (gpu_index + 1) % ${#gpus[@]} ))
          current_workers=$((current_workers + 1))

          if (( current_workers >= max_workers )); then
            wait -n
            current_workers=$((current_workers - 1))
          fi

          sleep 1
        done
      done
    done
  done
done

wait

echo "All tasks have completed."
