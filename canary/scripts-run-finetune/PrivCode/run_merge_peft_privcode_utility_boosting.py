import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


# gpus = ['2', '3', '4', '5']
gpus = ['7', '1', '2', '3', '4', '5', '6', '0']
# gpus = ['5', '6', '7']
# gpus = ['0', '1', '2', '3']
gpus = ['0']

dp_epsilons = [0.2, 1, 4, 10]
dp_epsilons = [0.2]
# dp_epsilons = [1, 4, 10]
dp_epsilons = ['inf']
# dp_epsilons = ['inf', 10]

lambda_kls = [1000]
# kl_steps = [5]
# kl_steps = [20, 30, 40]
kl_steps = [10000]

alpha=0.01  # the bigger, the more fastly lambda declines
max_lambda=100  # for ds-coder
# max_lambda=1
# max_lambda=5   # for starcoder7b

min_lambda=0.1

reps = [1, 10]

steps = [110, ]

# base_model = "bigcode/starcoder2-3b"
# base_model = "bigcode/starcoder2-7b"
base_model = "Qwen/Qwen2.5-Coder-1.5B"
# base_model = "Qwen/Qwen2.5-Coder-7B"
# base_model = "deepseek-ai/deepseek-coder-1.3b-base"
# base_model = "deepseek-ai/deepseek-coder-6.7b-base"

base_models = [
                "deepseek-ai/deepseek-coder-6.7b-base", 
               "google/codegemma-7b", 
               "Qwen/Qwen2.5-Coder-7B",
               "Qwen/CodeQwen1.5-7B"
               ]


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
    
    for base_model in base_models:
        for dp_epsilon in dp_epsilons:
            for step in steps:
                for kl_step in kl_steps:
                    for lambda_kl in lambda_kls:
                        for rep in reps:
                            model_name = base_model.split("/")[-1]
                            peft_model_path = f'.../checkpoints_codeonly/canary/{model_name}/dp{dp_epsilon}_baseline_rep{rep}/checkpoint-{step}'
                            output_path = f'.../checkpoints_codeonly/canary/{model_name}/dp{dp_epsilon}_baseline_rep{rep}_merged/checkpoint-{step}'
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
