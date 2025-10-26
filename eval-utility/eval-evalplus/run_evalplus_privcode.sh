#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
gpus=("5" "0" "2")
gpus=("0" "1")
# gpus=("2" "3" "4" "5" "6" "7")
# gpus=("4" "5")
# gpus=("0" "1" "2" )

# MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
# MODEL_PATH="google/codegemma-7b"
# MODEL_PATH="Qwen/CodeQwen1.5-7B"

MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
global_dp_epsilon=4
dp_epsilons=(4)
dp_epsilons=('inf')


steps=(2000)
steps=(100)
# steps=(200)
# steps=(100 )
# steps=( 800)

# round-trip configs
round_trip_model="Llama-3.1-70B-Instruct"
sim_thresholds=(0.82)

# ast configs
alphas=(0.01)
max_lambdas=(1000)

# data size
data_sizes=(55500)

# Static parameters
datasets=("humaneval" "mbpp")
datasets=("mbpp")
# datasets=("humaneval")

is_baseline='yes'
is_baseline='no'

backend="vllm"
# backend="hf"
tp=1
greedy="--greedy"

max_workers=2


# Initialize GPU index
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, and step
for dp_epsilon in "${dp_epsilons[@]}"; do
  for sim_threshold in "${sim_thresholds[@]}"; do
    for alpha in "${alphas[@]}"; do
      for max_lambda in "${max_lambdas[@]}"; do
        for data_size in "${data_sizes[@]}"; do
          for step in "${steps[@]}"; do
            for dataset in "${datasets[@]}"; do
              # Set the output path and checkpoint path based on current parameters
              # dp baseline
              output_root="generate/evalplus/${MODEL_NAME}/instruct"
              # output_root="generate/evalplus_0.3.1/${MODEL_NAME}/step2_codeonly/complete"
              if [ "$dp_epsilon" == "inf" ]; then
                  if [ "$is_baseline" == "no" ]; then
                      checkpoint_path=".../dpcode/checkpoints_codeonly/step2_promptsim_${round_trip_model}_tau${sim_threshold}/${MODEL_NAME}_dp${global_dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_datasize${data_size}/privsyn_merged/checkpoint-${step}"
                  else 
                      checkpoint_path=".../dpcode/checkpoints_codeonly/step2_promptsim_${round_trip_model}_tau${sim_threshold}/${MODEL_NAME}_dp${global_dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_datasize${data_size}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
                  fi
              else
                  checkpoint_path=".../dpcode/checkpoints_codeonly/step2_promptsim_${round_trip_model}_tau${sim_threshold}/${MODEL_NAME}_dp${global_dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_datasize${data_size}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
              fi

              # Define the command with parameters for evalplus.evaluate
              command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
                --model \"${checkpoint_path}\" \
                --root \"${output_root}\" \
                --dataset \"${dataset}\" \
                --backend \"${backend}\" \
                --force-base-prompt \
                --tp ${tp} \
                ${greedy}"
              
              # command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
              #   --model \"${checkpoint_path}\" \
              #   --root \"${output_root}\" \
              #   --dataset \"${dataset}\" \
              #   --backend \"${backend}\" \
              #   --force-base-prompt \
              #   --tp ${tp} \
              #   --temperature 1.0  \
              #   --n-samples 1 "


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
      done
    done
  done
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
