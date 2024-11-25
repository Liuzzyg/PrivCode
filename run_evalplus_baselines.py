import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['4', '5']
gpus = ['0', '1', '2', '3']

dp_epsilons = [10]
# dp_epsilons = ['inf']
# steps = [12]
# steps = [65]
# steps = [80, 100]
steps = [15, 16]
# steps = [30, 40, 50, 60, 35, 45, 55]

model = "bigcode/starcoder2-7b"
model = "bigcode/starcoder2-3b"
# model = "deepseek-ai/deepseek-coder-1.3b-base"
# model = "deepseek-ai/deepseek-coder-6.7b-base"
batch_size = 16

dataset = "mbpp"

is_pretrained = False   # run evalplus on pretrain model
is_baseline = True

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
            if is_pretrained:
                output_path = f"generate/evalplus/magicoder/{model_name}/pretrain/{dataset}/samples.jsonl"
                checkpoint_path = None
                arguments = [
                    '--checkpoint', model,
                    '--checkpoint_path', checkpoint_path,
                    '--output_path', output_path,
                    '--batch_size', batch_size,
                    '--is_pretrained',
                    '--dataset', dataset
                ]
            elif is_baseline:
                output_path = f"generate/evalplus/magicoder/{model_name}/dpbaseline/{dataset}/samples_dp{dp_epsilon}_step{step}.jsonl"
                checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/{model_name}/dp{dp_epsilon}_baseline_merged/checkpoint-{step}'

                arguments = [
                    '--checkpoint', model,
                    '--checkpoint_path', checkpoint_path,
                    '--output_path', output_path,
                    '--batch_size', batch_size,
                    '--is_baseline',
                    '--dataset', dataset
                ]
            else:
                print("pls check your config!!!")
                break
                
            script_path = 'evalplus.py'
            
            command = ['python', script_path] + [str(arg) for arg in arguments]

            futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

            gpu_index = (gpu_index + 1) % len(gpus)
            time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
