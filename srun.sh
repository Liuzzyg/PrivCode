#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 80G
#SBATCH -t 1-00:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1
#SBATCH -w jaguar01

nvidia-smi

conda activate dpcode_test
cd /p/fzv6enresearch/liuzheng/dpcode

bash run_finetune_astdp.sh