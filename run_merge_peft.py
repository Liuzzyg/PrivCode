import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['0', '1', '2', '3', '4', '5', '6', '7']
gpus = ['0', '1', '2', '3']
# gpus = ['4', '5', '6', '7']

dp_epsilons = [10]
# dp_epsilons = ['inf']

lambda_kl = 0.1
# kl_steps = [5]
kl_steps = [20, 30, 40]

steps = [75, 85, 95, 105, 115]
steps = [765, 755, 770, 790, 810, 840, 835, 830]
# steps = [505, 510, 515, 520, 495, 490, 485, 480]
steps = [90, 100, 110, 120]

# base_model = "bigcode/starcoder2-3b"
# base_model = "Qwen/Qwen2.5-Coder-1.5B"
# base_model = "deepseek-ai/deepseek-coder-6.7b-base"
base_model = "deepseek-ai/deepseek-coder-1.3b-base"


is_baseline = True
is_baseline = False

max_workers = 12


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
            for kl_step in kl_steps:
                model_name = base_model.split("/")[-1]
                if is_baseline:
                    peft_model_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_baseline/checkpoint-{step}'
                    output_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_baseline_merged/checkpoint-{step}'
                    arguments = [
                        '--base_model_name_or_path', base_model,
                        '--peft_model_path', peft_model_path,
                        '--save_merged_model_path', output_path,
                    ]
                else:
                    # pdb.set_trace()
                    peft_model_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_lambda{lambda_kl}_klstep{kl_step}/checkpoint-{step}'
                    output_path = f"/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_lambda{lambda_kl}_klstep{kl_step}_merged/checkpoint-{step}"
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
