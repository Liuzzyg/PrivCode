# import json

# def process_solution(solution):
#     # 查找第一个和第二个三引号的位置
#     first_triple_quote = solution.find('"""')
#     if first_triple_quote == -1:
#         return solution  # 如果没有找到三引号，返回原始 solution

#     second_triple_quote = solution.find('"""', first_triple_quote + 3)
#     if second_triple_quote == -1:
#         return solution  # 如果只有一个三引号，返回原始 solution

#     # 提取注释内容和剩余的代码
#     docstring = solution[first_triple_quote:second_triple_quote + 3]
#     remaining_code = solution[:first_triple_quote] + solution[second_triple_quote + 3:]

#     # 将注释移动到 solution 开头，后接一个换行符，再加上剩余代码
#     return f"{docstring}\n{remaining_code.strip()}"

# def process_jsonl_file(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             data = json.loads(line)
#             # 处理 solution 字段
#             data['solution'] = process_solution(data['solution'])
#             # 写入新的 jsonl 文件
#             outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# # 使用示例
# input_file = 'generate/evalplus_0.3.1/starcoder2-3b/dpbaseline/mbpp/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_code--magicoder--starcoder2-3b--dp10_baseline_merged--checkpoint-12_vllm_temp_0.0.raw.jsonl'   # 替换为你的输入文件名
# output_file = 'generate/evalplus_0.3.1/starcoder2-3b/dpbaseline/mbpp/manual_filter/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_code--magicoder--starcoder2-3b--dp10_baseline_merged--checkpoint-12_vllm_temp_0.0.filtered.jsonl' # 替换为你希望输出的文件名
# process_jsonl_file(input_file, output_file)










import json

def process_solution(solution):
    # 查找第一个和第二个三引号的位置
    first_triple_quote = solution.find('"""')
    if first_triple_quote == -1:
        return solution  # 如果没有找到三引号，返回原始 solution

    second_triple_quote = solution.find('"""', first_triple_quote + 3)
    if second_triple_quote == -1:
        return solution  # 如果只有一个三引号，返回原始 solution

    # 删除第一个和第二个三引号之间的内容，包括这两个引号
    remaining_code = solution[:first_triple_quote] + solution[second_triple_quote + 3:]
    return remaining_code.strip()  # 删除前后多余空格

def process_jsonl_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            # 处理 solution 字段
            data['solution'] = process_solution(data['solution'])
            # 写入新的 jsonl 文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

# 使用示例
input_file = 'generate/evalplus_0.3.1/starcoder2-3b/dpbaseline/mbpp/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_code--magicoder--starcoder2-3b--dp10_baseline_merged--checkpoint-12_vllm_temp_0.0.raw.jsonl'   # 替换为你的输入文件名
output_file = 'generate/evalplus_0.3.1/starcoder2-3b/dpbaseline/mbpp/manual_filter/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_code--magicoder--starcoder2-3b--dp10_baseline_merged--checkpoint-12_vllm_temp_0.0.delete_filtered.jsonl' # 替换为你希望输出的文件名
process_jsonl_file(input_file, output_file)
