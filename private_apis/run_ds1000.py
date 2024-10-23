import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['0', '1']
# gpus = ['3']
gpus = ['0', '1', '2', '3']

dp_epsilons = [10]
steps = [520, 570, 625]
# steps = [565, 640, 580, 655, 595, 610, 670, 625]
# steps = [90, 110]
# steps = [95, 105, 480, 400]

# model = "deepseek-ai/deepseek-coder-6.7b-base"
model = "deepseek-ai/deepseek-coder-1.3b-instruct"
batch_size = 12

is_post_step = False   # true for step2
is_pretrained = False   # run evalplus on pretrain model

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
                if not is_pretrained:
                    if not is_post_step:
                        # output_path = f"generate/ds1000/private_api_numpy/step1/{model_name}-dp{dp_epsilon}_lbs512_codesnippet_step{step}-answers.jsonl"
                        # checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/private_api_numpy/{model_name}/dp{dp_epsilon}_lbs512_codesnippet/checkpoint-{step}'

                        output_path = f"generate/ds1000/private_api_numpy/step1/{model_name}-dp{dp_epsilon}_lbs32_step{step}-answers.jsonl"
                        checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/private_api_numpy/{model_name}/dp{dp_epsilon}_lbs32/checkpoint-{step}'
                        arguments = [
                            '--model', model,
                            '--checkpoint_path', checkpoint_path,
                            '--output_path', output_path,
                            '--batch_size', batch_size
                        ]
                    else:
                        output_path = f"generate/ds1000/private_api_numpy/step2/{model_name}-dp{dp_epsilon}_step{step}-answers.jsonl"
                        checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/private_api_numpy/{model_name}/dp{dp_epsilon}_syndata/checkpoint-{step}'
                        arguments = [
                            '--checkpoint', model,
                            '--checkpoint_path', checkpoint_path,
                            '--output_path', output_path,
                            '--batch_size', batch_size,
                            # '--is_post_step'
                        ]
                else:
                    output_path = f"generate/ds1000/private_api_numpy/pretrain/{model_name}-answers.jsonl"
                    checkpoint_path = None
                    arguments = [
                        '--checkpoint', model,
                        # '--checkpoint_path', checkpoint_path,
                        '--output_path', output_path,
                        '--batch_size', batch_size,
                        '--is_pretrained'
                    ]
                    
                script_path = 'DS-1000/run_dist_inference.py'
                
                command = ['python', script_path] + [str(arg) for arg in arguments]

                futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                gpu_index = (gpu_index + 1) % len(gpus)
                time.sleep(1) # especially for fine-tuning

    concurrent.futures.wait(futures)
