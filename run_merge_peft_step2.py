import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


# gpus = ['2', '3', '4', '5']
gpus = ['0', '1', '2', '3']
# gpus = ['5', '6', '7']
# gpus = ['3', '2']
gpus = ['4', '5', '6', '7']

dp_epsilons = [1]
# dp_epsilons = ['inf']
# dp_epsilons = ['inf', 10]


steps = [950, 800, 600, 400]
steps = [110, 90, 125, 130, 140, 160, 170, 190]
# steps = [765, 755, 770, 790, 810, 840, 835, 830]
# steps = [16, 20, 25, 28, 30, 32, 35, 50, 60, 64, 70, 80]
# steps = [600, 700, 800, 900, 1000, 1200, 1300, 750]
steps = [50, 200]
# steps = [1100, 1300, 1500, 1700, 1200, 1400, 1600, 1000]

# base_model = "bigcode/starcoder2-3b"
# base_model = "bigcode/starcoder2-7b"
# base_model = "Qwen/Qwen2.5-Coder-1.5B"
# base_model = "Qwen/Qwen2.5-Coder-7B"
# base_model = "deepseek-ai/deepseek-coder-1.3b-base"
base_model = "deepseek-ai/deepseek-coder-6.7b-base"


max_workers = 8


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
            if dp_epsilon == 'inf':
                peft_model_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2/{model_name}/dpinf/checkpoint-{step}'
                output_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2/{model_name}/dpinf_merged/checkpoint-{step}'
                arguments = [
                    '--base_model_name_or_path', base_model,
                    '--peft_model_path', peft_model_path,
                    '--save_merged_model_path', output_path,
                    ]
            else:
                peft_model_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2/{model_name}/dp{dp_epsilon}_baseline/checkpoint-{step}'
                output_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/step2/{model_name}/dp{dp_epsilon}_baseline_merged/checkpoint-{step}'
                arguments = [
                    '--base_model_name_or_path', base_model,
                    '--peft_model_path', peft_model_path,
                    '--save_merged_model_path', output_path,
                    ]
            script_path = 'examples/codegen/finetune/merge_peft_adapters.py'
            
            command = ['python', script_path] + [str(arg) for arg in arguments]

            futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

            gpu_index = (gpu_index + 1) % len(gpus)
            time.sleep(5) # especially for fine-tuning

    concurrent.futures.wait(futures)
