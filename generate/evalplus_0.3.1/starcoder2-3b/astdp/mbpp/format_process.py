import json
import ast
import astor

def format_solution_code(code):
    """格式化代码字符串以修复缩进"""
    try:
        # 将代码解析为 AST
        tree = ast.parse(code)
        # 使用 astor 将 AST 转换回代码字符串，修复缩进
        formatted_code = astor.to_source(tree)
        return formatted_code
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        return "no"  # 如果格式化失败，则返回原始代码

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析 JSON 数据
            data = json.loads(line)
            if "solution" in data:
                # 格式化 solution 部分代码
                data["solution"] = format_solution_code(data["solution"])
            # 写入新 JSONL 文件
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


# 使用示例
input_file = "/p/fzv6enresearch/liuzheng/dpcode/generate/evalplus_0.3.1/starcoder2-3b/astdp/mbpp/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_code--magicoder--starcoder2-3b--dp10_lambda1to0.1_alpha0.01_merged--checkpoint-12_vllm_temp_0.0.raw.jsonl"  # 输入的 JSONL 文件
output_file = "/p/fzv6enresearch/liuzheng/dpcode/generate/evalplus_0.3.1/starcoder2-3b/astdp/mbpp_formatted/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_code--magicoder--starcoder2-3b--dp10_lambda1to0.1_alpha0.01_merged--checkpoint-12_vllm_temp_0.0.raw.jsonl"  # 输出的 JSONL 文件
process_jsonl(input_file, output_file)
