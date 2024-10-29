import torch

# 读取 .bin 文件
file_path = '/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/Qwen2.5-Coder-1.5B/dp10_lambda0.05_klstep5/checkpoint-90/training_args.bin'
training_args = torch.load(file_path)

# 查看内容
print(training_args)
