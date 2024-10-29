#!/bin/bash

#SBATCH -c 32
#SBATCH --mem 80G
#SBATCH -t 1-00:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 1

export PYTHONPATH=/u/fzv6en/anaconda3/envs/dpcode_test/bin/python

nvidia-smi

$PYTHONPATH train_latent_classifier.py