import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['0', '1', '2', '3']
# gpus = ['0', '1']

dp_epsilons = [10]

MAX_LAMBDA = 1
MIN_LAMBDA = 0.1
ALPHA = 0.01

steps = [210, 220, 230, 240]
steps = [30, 40, 50, 60, 35, 45, 55]
steps = [1, 2, 3, 4]

model = "deepseek-ai/deepseek-coder-6.7b-base"
model = "bigcode/starcoder2-3b"
# model = "deepseek-ai/deepseek-coder-1.3b-instruct"

dataset = 'mbpp'

batch_size = 16

max_workers = 4


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
            # pdb.set_trace()
            model_name = model.split("/")[-1]

            output_path = f"generate/evalplus/magicoder/{model_name}/astdp/{dataset}/samples_dp{dp_epsilon}_lambda{MAX_LAMBDA}to{MIN_LAMBDA}_alpha{ALPHA}_step{step}.jsonl"
            checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_lambda{MAX_LAMBDA}to{MIN_LAMBDA}_alpha{ALPHA}/checkpoint-{step}'
            arguments = [
                '--checkpoint', model,
                '--checkpoint_path', checkpoint_path,
                '--output_path', output_path,
                '--batch_size', batch_size,
                '--dataset', dataset
            ]

            script_path = 'evalplus.py'
            
            command = ['python', script_path] + [str(arg) for arg in arguments]

            futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

            gpu_index = (gpu_index + 1) % len(gpus)
            time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
