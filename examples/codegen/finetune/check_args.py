import torch

# 文件路径
checkpoint_path = "/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/magicoder/deepseek-coder-1.3b-instruct/dp10/checkpoint-40/training_args.bin"

# 加载 training_args.bin 文件
training_args = torch.load(checkpoint_path)

# 打印出训练参数
print(training_args)
