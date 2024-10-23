import torch

# 读取 .bin 文件
file_path = '/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/synthetic_numpy/deepseek-coder-1.3b-instruct/dp10_nolora_1norm/checkpoint-0/training_args.bin'
training_args = torch.load(file_path)

# 查看内容
print(training_args)
