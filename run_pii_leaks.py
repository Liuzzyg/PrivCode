import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['1', '2']

dp_epsilons = [10]
steps = [130]

model = "deepseek-ai/deepseek-coder-6.7b-base"

is_private_syndata_step2s = ['yes', 'no']   # 'yes' for model finetuned on private syndata, while 'no' for original data
is_pretrained = False   # run evalplus on pretrain model

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
            for is_private_syndata_step2 in is_private_syndata_step2s:
                is_private_syndata_step2 = is_private_syndata_step2 in ['yes', 'y']
                # pdb.set_trace()
                model_name = model.split("/")[-1]
                if not is_pretrained:
                    if is_private_syndata_step2:
                        output_path = f"pii_leakage/queries/{model_name}_dp{dp_epsilon}_syndata_step{step}.jsonl"
                        checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dp{dp_epsilon}_syndata/checkpoint-{step}'
                        arguments = [
                            '--base_checkpoint', model,
                            '--model_checkpoint', checkpoint_path,
                            '--output_file', output_path
                        ]
                    else:
                        output_path = f"pii_leakage/queries/{model_name}_original_data_step{step}.jsonl"
                        checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/original_data/checkpoint-{step}'
                        arguments = [
                            '--base_checkpoint', model,
                            '--model_checkpoint', checkpoint_path,
                            '--output_file', output_path
                        ]
                else:
                    output_path = f"pii_leakage/queries/{model_name}_pretrained.jsonl"
                    checkpoint_path = None
                    arguments = [
                        '--base_checkpoint', model,
                        '--model_checkpoint', checkpoint_path,
                        '--output_file', output_path
                    ]
                    
                script_path = 'pii_leaks_copy.py'
                
                command = ['python', script_path] + [str(arg) for arg in arguments]

                futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                gpu_index = (gpu_index + 1) % len(gpus)
                time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
