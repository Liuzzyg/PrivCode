import json
import random

# 输入文件和输出文件路径
input_file = 'data/private_syn/Qwen2.5-Coder-1.5B/private_syndata_55k_dp10.jsonl'
output_file = 'data/private_syn/Qwen2.5-Coder-1.5B/private_syndata_55k_dp10_test.jsonl'

# 读取输入文件中的所有样本
with open(input_file, 'r', encoding='utf-8') as infile:
    samples = [json.loads(line) for line in infile]

# 随机选择100个样本
random_samples = random.sample(samples, 2000)

# 将随机选中的样本写入输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for sample in random_samples:
        json.dump(sample, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"已成功将100个随机样本保存到 {output_file}")
