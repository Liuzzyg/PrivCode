import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['0', '1', '2', '3']
# gpus = ['3']

dp_epsilons = [10]
steps = [20, 25, 30]

# model = "deepseek-ai/deepseek-coder-6.7b-base"
model = "deepseek-ai/deepseek-coder-1.3b-instruct"
batch_size = 16

is_post_step = False   # true for step2
is_private_syndata_step2s = ['yes', 'no']   # 'yes' for model finetuned on private syndata, while 'no' for original data
is_private_syndata_step2s = ['no']
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
            for is_private_syndata_step2 in is_private_syndata_step2s:
                is_private_syndata_step2 = is_private_syndata_step2 in ['yes', 'y']
                # pdb.set_trace()
                model_name = model.split("/")[-1]
                if not is_pretrained:
                    if not is_post_step:
                        output_path = f"generate/evalplus/magicoder/step1/samples_{model_name}_dp{dp_epsilon}_test_step{step}.jsonl"
                        checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/magicoder/{model_name}/dp{dp_epsilon}_test/checkpoint-{step}'
                        arguments = [
                            '--checkpoint', model,
                            '--checkpoint_path', checkpoint_path,
                            '--output_path', output_path,
                            '--batch_size', batch_size
                        ]
                    else:
                        if is_private_syndata_step2:
                            output_path = f"generate/evalplus/magicoder/step2/samples_{model_name}_dp{dp_epsilon}_filtered_syndata_step{step}.jsonl"
                            checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dp{dp_epsilon}_syndata/checkpoint-{step}'
                            arguments = [
                                '--checkpoint', model,
                                '--checkpoint_path', checkpoint_path,
                                '--output_path', output_path,
                                '--batch_size', batch_size,
                                # '--is_post_step'
                            ]
                        else:
                            output_path = f"generate/evalplus/magicoder/step2/samples_{model_name}_filtered_original_data_step{step}.jsonl"
                            checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/original_data/checkpoint-{step}'
                            arguments = [
                                '--checkpoint', model,
                                '--checkpoint_path', checkpoint_path,
                                '--output_path', output_path,
                                '--batch_size', batch_size,
                                # '--is_post_step'
                            ]
                else:
                    output_path = f"generate/evalplus/pretrained_model/samples_{model_name}.jsonl"
                    checkpoint_path = None
                    arguments = [
                        '--checkpoint', model,
                        '--checkpoint_path', checkpoint_path,
                        '--output_path', output_path,
                        '--batch_size', batch_size,
                        '--is_pretrained'
                    ]
                    
                script_path = 'evalplus.py'
                
                command = ['python', script_path] + [str(arg) for arg in arguments]

                futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                gpu_index = (gpu_index + 1) % len(gpus)
                time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
