import torch

# 读取 .bin 文件
file_path = "/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/Qwen2.5-Coder-1.5B/dp10_lambda1000to0.1_alpha0.01/checkpoint-2/training_args.bin"
training_args = torch.load(file_path)

# 查看内容
print(training_args)
