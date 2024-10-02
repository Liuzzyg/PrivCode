import json
import random
from tqdm import tqdm
import pdb

def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def extract_numpy_prompts(data):
    prompts = []
    context_size = 7

    for sample in tqdm(data, desc="Processing samples"):
        content = sample['content']
        lines = content.split('\n')

        # 找出所有包含 'np.' 的行号
        np_lines = [i for i, line in enumerate(lines) if 'np.' in line]

        # 随机打乱np函数所在行的顺序
        random.shuffle(np_lines)

        # 检查每个np行是否有前后7行
        found = False
        for i in np_lines:
            if i - context_size >= 0 and i + context_size < len(lines):
                # 提取前后7行
                start = i - context_size
                end = i + context_size + 1
                selected_code = '\n'.join(lines[start:end])
                # pdb.set_trace()
                # 保存到prompts中
                prompts.append({
                    'index': sample['index'],
                    'content': selected_code
                })
                found = True
                break  # 一旦找到符合条件的，跳出循环

        # 如果没有找到满足条件的np函数，则跳过该样本
        if not found:
            continue
    
    return prompts

# 主函数
def main(input_path, output_path):
    data = load_jsonl(input_path)
    
    prompts = extract_numpy_prompts(data)
    
    save_jsonl(output_path, prompts)

input_path = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_subset25k.jsonl'
output_path = 'data/oss_instruction/random_numpy_code_snippet.jsonl'
input_path = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_test.jsonl'
output_path = 'data/oss_instruction/random_numpy_code_snippet_test.jsonl'
main(input_path, output_path)
