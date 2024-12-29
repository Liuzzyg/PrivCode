#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
gpus=("1" "0" "2" "3")
# gpus=("1")
# gpus=("2" "3" "4" "5" "6" "7")
# gpus=("4" "5")
gpus=("0"  )

MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
# MODEL_PATH="google/codegemma-7b"

# MODEL_PATHS=("google/codegemma-7b" "Qwen/CodeQwen1.5-7B" )
# MODEL_PATHS=("Qwen/CodeQwen1.5-7B")
# MODEL_PATHS=("google/codegemma-7b")
MODEL_PATHS=("deepseek-ai/deepseek-coder-6.7b-base" "Qwen/Qwen2.5-Coder-7B" "Qwen/CodeQwen1.5-7B" "google/codegemma-7b")

global_dp_epsilon=10
dp_epsilons=(4)
dp_epsilons=('inf')

steps=(2000)
steps=(100 200)
# steps=(200)
# steps=(100 )

# round-trip configs
round_trip_model="Llama-3.1-70B-Instruct"
sim_thresholds=(0.88)

# ast configs
alphas=(0.01)
max_lambdas=(
            # 1 
            1000 
            # 100000
            )

# data size
data_sizes=(55500)

# Static parameters
datasets=("humaneval" "mbpp")
# datasets=("mbpp")
# datasets=("humaneval")

is_baseline='no'

backend="vllm"
# backend="hf"
tp=1
greedy="--greedy"

max_workers=1

chat_model='y'
# chat_model='n'

# Initialize GPU index
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, and step
for dataset in "${datasets[@]}"; do
  for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for dp_epsilon in "${dp_epsilons[@]}"; do
      for sim_threshold in "${sim_thresholds[@]}"; do
        for alpha in "${alphas[@]}"; do
          for max_lambda in "${max_lambdas[@]}"; do
            for data_size in "${data_sizes[@]}"; do
              for step in "${steps[@]}"; do
              
                MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

                if [ "$chat_model" == "y" ]; then
                    output_root="generate/evalplus_0.3.1/hyper_parameter_analysis/epsilon/chat/${MODEL_NAME}"
                else
                    output_root="generate/evalplus_0.3.1/hyper_parameter_analysis/epsilon/completion/${MODEL_NAME}"
                fi
                
                # Set the output path and checkpoint path based on current parameters
                # dp baseline
                if [ "$dp_epsilon" == "inf" ]; then
                    if [ "$is_baseline" == "no" ]; then
                        checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/hyper_parameter_analysis/step2_promptsim_${round_trip_model}_tau${sim_threshold}/${MODEL_NAME}_dp${global_dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_datasize${data_size}/privsyn_merged/checkpoint-${step}"
                    else 
                        checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/hyper_parameter_analysis/step2_promptsim_${round_trip_model}_tau${sim_threshold}/${MODEL_NAME}_dp${global_dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_datasize${data_size}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
                    fi
                else
                    checkpoint_path="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/hyper_parameter_analysis/step2_promptsim_${round_trip_model}_tau${sim_threshold}/${MODEL_NAME}_dp${global_dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_datasize${data_size}/dp${dp_epsilon}_baseline_merged/checkpoint-${step}"
                fi

                # Define the command with parameters for evalplus.evaluate
                if [ "$chat_model" == "y" ]; then
                    command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
                      --model \"${checkpoint_path}\" \
                      --root \"${output_root}\" \
                      --dataset \"${dataset}\" \
                      --backend \"${backend}\" \
                      --tp ${tp} \
                      ${greedy}"
                else
                    command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} evalplus.evaluate \
                      --model \"${checkpoint_path}\" \
                      --root \"${output_root}\" \
                      --dataset \"${dataset}\" \
                      --backend \"${backend}\" \
                      --tp ${tp} \
                      --force-base-prompt \
                      ${greedy}"
                fi

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
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
