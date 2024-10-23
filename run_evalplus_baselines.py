import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['3', '1', '2']

dp_epsilons = [10]
# steps = [130]
steps = [50, 60, 70]

model = "deepseek-ai/deepseek-coder-6.7b-base"
batch_size = 8

is_pretrained = False   # run evalplus on pretrain model
is_baseline = True

max_workers = 20


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
                output_path = f"generate/evalplus/pretrained_model/samples_{model_name}.jsonl"
                checkpoint_path = None
                arguments = [
                    '--checkpoint', model,
                    '--checkpoint_path', checkpoint_path,
                    '--output_path', output_path,
                    '--batch_size', batch_size,
                    '--is_pretrained'
                ]
            elif is_baseline:
                output_path = f"generate/evalplus/magicoder/dpsgd_baseline/samples_{model_name}_step{step}.jsonl"
                checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dpsgd_baseline_merged/checkpoint-{step}'
                arguments = [
                    '--checkpoint', model,
                    '--checkpoint_path', checkpoint_path,
                    '--output_path', output_path,
                    '--batch_size', batch_size,
                    '--is_baseline'
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
