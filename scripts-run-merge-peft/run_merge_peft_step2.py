import os
import glob
import concurrent.futures
import csv
import time
import subprocess
import pdb


# gpus = ['2', '3', '4', '5']
gpus = ['0', '1', '2', '3']
# gpus = ['2', '3', '4', '5', '6', '7']
# gpus = ['0', '1', '2']
# gpus = ['2']
# gpus = ['0', '1']
# gpus = ['5', '5']

global_dp_epsilons = 4
dp_epsilons = [4]
# dp_epsilons = ['inf']
# dp_epsilons = ['inf', 10]


steps = [2000]
steps = [100]
steps = [ 800]


# base_model = "bigcode/starcoder2-3b"
base_model = "bigcode/starcoder2-7b"
# base_model = "Qwen/Qwen2.5-Coder-1.5B"
base_model = "Qwen/Qwen2.5-Coder-7B"
# base_model = "deepseek-ai/deepseek-coder-1.3b-base"
base_model = "deepseek-ai/deepseek-coder-6.7b-base"
# base_model = "google/codegemma-7b"
base_model = "Qwen/CodeQwen1.5-7B"

# round-trip configs
round_trip_model = "Llama-3.1-70B-Instruct"
sim_thresholds = [0.82]

# ast configs
alphas = [0.01]
max_lambdas = [1000]

# data size
data_sizes = [55500]

is_baseline = False
is_baseline = True

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
    
    for dp_epsilon in dp_epsilons:
        for step in steps:
            for sim_threshold in sim_thresholds:
                for alpha in alphas:
                    for max_lambda in max_lambdas:
                        for data_size in data_sizes:
                            model_name = base_model.split("/")[-1]
                            if dp_epsilon == 'inf':
                                if not is_baseline:
                                    peft_model_path = f'.../checkpoints_codeonly/step2_promptsim_{round_trip_model}_tau{sim_threshold}/{model_name}_dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_datasize{data_size}/privsyn/checkpoint-{step}'
                                    output_path = f'.../checkpoints_codeonly/step2_promptsim_{round_trip_model}_tau{sim_threshold}/{model_name}_dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_datasize{data_size}/privsyn_merged/checkpoint-{step}'
                                    arguments = [
                                        '--base_model_name_or_path', base_model,
                                        '--peft_model_path', peft_model_path,
                                        '--save_merged_model_path', output_path,
                                        ]
                                else:
                                    peft_model_path = f'.../checkpoints_codeonly/step2_promptsim_{round_trip_model}_tau{sim_threshold}/{model_name}_dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_datasize{data_size}/dpinf_baseline/checkpoint-{step}'
                                    output_path = f'.../checkpoints_codeonly/step2_promptsim_{round_trip_model}_tau{sim_threshold}/{model_name}_dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_datasize{data_size}/dpinf_baseline_merged/checkpoint-{step}'
                                    arguments = [
                                        '--base_model_name_or_path', base_model,
                                        '--peft_model_path', peft_model_path,
                                        '--save_merged_model_path', output_path,
                                        ]
                            else:
                                peft_model_path = f'.../checkpoints_codeonly/step2_promptsim_{round_trip_model}_tau{sim_threshold}/{model_name}_dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_datasize{data_size}/dp{dp_epsilon}_baseline/checkpoint-{step}'
                                output_path = f'.../checkpoints_codeonly/step2_promptsim_{round_trip_model}_tau{sim_threshold}/{model_name}_dp{global_dp_epsilons}_lambda{max_lambda}to0.1_alpha{alpha}_datasize{data_size}/dp{dp_epsilon}_baseline_merged/checkpoint-{step}'
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
