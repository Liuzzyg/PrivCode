#!/bin/bash

gpus=("0" "1")

MODEL_PATHs=( "Qwen/Qwen2.5-Coder-7B" )
dp_epsilons=(4)
steps=(100)

# round-trip configs
round_trip_model="Llama-3.1-70B-Instruct"
sim_thresholds=(0.88)

# ast configs
alpha=0.01
max_lambdas=(1000)

# Static parameters
datasets=("full" "hard")

# 'instruct' means instruction following task, 'complete' means code completion task
splits=('instruct' 'complete')

backend="vllm"
tp=1
BATCH_SIZE=1
greedy="--greedy"
resume="--resume"

max_workers=2

gpu_index=0
current_workers=0

for MODEL_PATH in "${MODEL_PATHs[@]}"; do
  for dp_epsilon in "${dp_epsilons[@]}"; do
    for split in "${splits[@]}"; do
      for step in "${steps[@]}"; do
        for dataset in "${datasets[@]}"; do
          for sim_threshold in "${sim_thresholds[@]}"; do
            for max_lambda in "${max_lambdas[@]}"; do
              MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')
              output_root="results/privcode/${MODEL_NAME}/bigcodebench/${split}/${dataset}"
              checkpoint_path="checkpoints/privcode/utility_boosting/${MODEL_NAME}/${round_trip_model}_tau${sim_threshold}/dp${dp_epsilon}_lambda${max_lambda}to0.1_alpha${alpha}_merged/checkpoint-${step}"
          
              command="CUDA_VISIBLE_DEVICES=${gpus[$gpu_index]} bigcodebench.generate \
                --model \"${checkpoint_path}\" \
                --split \"${split}\" \
                --subset \"${dataset}\" \
                --bs \"${BATCH_SIZE}\" \
                --backend \"${backend}\" \
                --save_root_path \"${output_root}\" \
                ${greedy} ${resume}" 

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
  done
done

wait

echo "All tasks have completed."
