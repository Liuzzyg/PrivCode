import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


gpus = ['0', '1']

dp_epsilons = [10]
steps = [130]

base_model = "deepseek-ai/deepseek-coder-6.7b-base"

is_private_syndata_step2s = ['yes', 'no']   # 'yes' for model finetuned on private syndata, while 'no' for original data
# is_private_syndata_step2s = ['yes']

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
                model_name = base_model.split("/")[-1]

                if is_private_syndata_step2:
                    peft_model_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dp{dp_epsilon}_syndata/checkpoint-{step}'
                    output_path = f"/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dp{dp_epsilon}_syndata_merged/checkpoint-{step}"
                    arguments = [
                        '--base_model_name_or_path', base_model,
                        '--peft_model_path', peft_model_path,
                        '--save_merged_model_path', output_path,
                    ]
                else:
                    checkpoint_path = f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/original_data/checkpoint-{step}'
                    output_path = f"/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/original_data_merged/checkpoint-{step}"
                    arguments = [
                        '--base_model_name_or_path', base_model,
                        '--peft_model_path', peft_model_path,
                        '--save_merged_model_path', output_path,
                    ]
                    
                script_path = 'examples/codegen/finetune/merge_peft_adapters.py'
                
                command = ['python', script_path] + [str(arg) for arg in arguments]

                futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                gpu_index = (gpu_index + 1) % len(gpus)
                time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
