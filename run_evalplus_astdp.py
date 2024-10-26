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
lambda_kl = [0.05]
kl_step = 5
steps = [150, 160, 175, 185]
# steps = [65]

model = "deepseek-ai/deepseek-coder-6.7b-base"
# model = "deepseek-ai/deepseek-coder-1.3b-instruct"
batch_size = 16

is_post_step = False   # true for step2

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
        for lam in lambda_kl:
            for step in steps:
                # pdb.set_trace()
                model_name = model.split("/")[-1]
                output_path = f"generate/evalplus/magicoder/astdp/samples_{model_name}_dp{dp_epsilon}_lambda{lam}_klstep{kl_step}_step{step}.jsonl"
                checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_lambda{lam}_klstep{kl_step}/checkpoint-{step}'
                arguments = [
                    '--checkpoint', model,
                    '--checkpoint_path', checkpoint_path,
                    '--output_path', output_path,
                    '--batch_size', batch_size
                ]

                script_path = 'evalplus.py'
                
                command = ['python', script_path] + [str(arg) for arg in arguments]

                futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                gpu_index = (gpu_index + 1) % len(gpus)
                time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
