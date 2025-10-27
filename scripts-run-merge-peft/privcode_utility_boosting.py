import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb

gpus = ['0', '1', '2', '3']

global_dp_epsilons = 4
dp_epsilons = ['inf']

steps = [1]

base_model = "Qwen/Qwen2.5-Coder-7B"

# round-trip configs
round_trip_model = "Llama-3.1-70B-Instruct"
round_trip_model = "Llama-3.1-8B-Instruct"
sim_thresholds = [0.88]

# ast configs
alphas = [0.01]
max_lambdas = [1000]

max_workers = 1

def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    subprocess.run(command)
    time.sleep(1)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0
    
    for dp_epsilon in dp_epsilons:
        for step in steps:
            for sim_threshold in sim_thresholds:
                for alpha in alphas:
                    for max_lambda in max_lambdas:
                        model_name = base_model.split("/")[-1]
                        peft_model_path = f'checkpoints/privcode/utility_boosting/{model_name}/{round_trip_model}_tau{sim_threshold}/dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}/checkpoint-{step}'
                        output_path = f'checkpoints/privcode/utility_boosting/{model_name}/{round_trip_model}_tau{sim_threshold}/dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_merged/checkpoint-{step}'
                        arguments = [
                            '--base_model_name_or_path', base_model,
                            '--peft_model_path', peft_model_path,
                            '--save_merged_model_path', output_path,
                            ]

                        script_path = 'examples/codegen/finetune/merge_peft_adapters.py'
                        
                        command = ['python', script_path] + [str(arg) for arg in arguments]

                        futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                        gpu_index = (gpu_index + 1) % len(gpus)
                        time.sleep(5)

    concurrent.futures.wait(futures)
