#!/bin/bash

# Define parameters
# gpus=("4" "5" "6" "7")
gpus=("0" "2" "3" "5" "4")
gpus=("2" "3" )
gpus=("0" "2")
# gpus=("4" "5" "6" "7" )

MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
# MODEL_PATH="bigcode/starcoder2-3b"
# MODEL_PATH="bigcode/starcoder2-7b"
# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-base"
# MODEL_PATH="Qwen/Qwen2.5-Coder-1.5B"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B"
MODEL_PATHs=("deepseek-ai/deepseek-coder-6.7b-base" "Qwen/Qwen2.5-Coder-7B" "Qwen/CodeQwen1.5-7B" "google/codegemma-7b")

MODEL_PATHs=( "Qwen/Qwen2.5-Coder-7B" )
# MODEL_PATHs=( "Qwen/CodeQwen1.5-7B" )
# MODEL_PATHs=(  "google/codegemma-7b")

dp_epsilons=(1 4)
dp_epsilons=(1 )


steps=(100)

# Static parameters
datasets=("full" "hard")
datasets=("full")
# datasets=("hard")

split='complete'
# split='instruct'
splits=('instruct' 'complete')
splits=('instruct')

backend="vllm"
# backend="hf"
tp=1
BATCH_SIZE=1
greedy="--greedy"
resume="--resume"


max_workers=2


# Initialize GPU index and process counter
gpu_index=0
current_workers=0

# Run evaluation for each combination of dp_epsilon, lambda_kl, and step
for MODEL_PATH in "${MODEL_PATHs[@]}"; do
  for dp_epsilon in "${dp_epsilons[@]}"; do
    for split in "${splits[@]}"; do
      for step in "${steps[@]}"; do
        for dataset in "${datasets[@]}"; do
          # declined lambda
          
          MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
          output_root="generate/bigcodebench/magicoder/${split}/${dataset}/${MODEL_NAME}/privsyn"
          checkpoint_path=".../checkpoints_codeonly/step2_promptsim_Llama-3.1-70B-Instruct_tau0.82/${MODEL_NAME}_dp${dp_epsilon}_lambda1000to0.1_alpha0.01_datasize55500/privsyn_merged/checkpoint-${step}"
      

          # Define the command with parameters for evalplus.evaluate
          command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} bigcodebench.generate \
            --model \"${checkpoint_path}\" \
            --split \"${split}\" \
            --subset \"${dataset}\" \
            --bs \"${BATCH_SIZE}\" \
            --backend \"${backend}\" \
            --save_root_path \"${output_root}\" \
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
  done
done

# Wait for all background processes to complete
wait

echo "All tasks have completed."
