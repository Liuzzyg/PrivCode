#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 80G
#SBATCH -t 1-00:00:00
#SBATCH -p gpu --gres=gpu:4
#SBATCH -n 1
#SBATCH -w jaguar01


nvidia-smi

source /u/fzv6en/anaconda3/etc/profile.d/conda.sh
cd /p/fzv6enresearch/liuzheng/dpcode


# conda activate evalplus

# bash run_evalplus_0.3.1_step2.sh



# conda activate dpcode_test

# python run_merge_peft_step2.py


conda activate dpcode_test

bash baseline/jft/finetune/run_finetune_step2_dp.sh



# conda activate dpcode_test

# bash run_finetune_step2_infbaseline.sh



# conda activate dpcode_test

# bash run_finetune_astdp.sh



# conda activate codebertscore

# bash data/private_syn/run_rt_test_prompt.sh