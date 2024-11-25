import torch

# 读取 .bin 文件
file_path = "/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/starcoder2-7b/dp10_baseline/checkpoint-155/training_args.bin"
training_args = torch.load(file_path)

# 查看内容
print(training_args)
