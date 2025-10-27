import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb

gpus = ['0', '1', '2']

dp_epsilons = [4]

alpha=0.01  # the bigger, the more fastly lambda declines
max_lambda=1000
min_lambda=0.1

steps = [1]
base_model = "Qwen/Qwen2.5-Coder-1.5B"
is_baseline = False

max_workers = 10

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
            model_name = base_model.split("/")[-1]
            if is_baseline:
                peft_model_path = f'checkpoints/privcode/privacy_sanitizing/{model_name}/dp{dp_epsilon}_baseline/checkpoint-{step}'
                output_path = f'checkpoints/privcode/privacy_sanitizing/{model_name}/dp{dp_epsilon}_baseline_merged/checkpoint-{step}'
                arguments = [
                    '--base_model_name_or_path', base_model,
                    '--peft_model_path', peft_model_path,
                    '--save_merged_model_path', output_path,
                ]
            else:
                peft_model_path = f'checkpoints/privcode/privacy_sanitizing/{model_name}/dp{dp_epsilon}_lambda{max_lambda}to{min_lambda}_alpha{alpha}/checkpoint-{step}'
                output_path = f'checkpoints/privcode/privacy_sanitizing/{model_name}/dp{dp_epsilon}_lambda{max_lambda}to{min_lambda}_alpha{alpha}_merged/checkpoint-{step}'
                
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
