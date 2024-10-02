import json

# 输入和输出文件路径
input_file = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_subset25k.jsonl'
output_file = 'data/starcoderdata_numpy_subset25k.jsonl'
input_file = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_test.jsonl'
output_file = 'data/starcoderdata_numpy_test.jsonl'

# 处理数据
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 解析 JSON Lines 文件中的每一行
        data = json.loads(line)
        # 仅保留 'content' 字段
        cleaned_data = {'content': data.get('content')}
        # 将处理后的数据写入新的文件
        outfile.write(json.dumps(cleaned_data) + '\n')

print(f"Cleaned data has been written to {output_file}")
