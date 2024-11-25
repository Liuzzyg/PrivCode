import json
import ast

input_file_path = "data/oss_instruction/processed_instruction_data_25k.jsonl"
output_file_path = "data/oss_instruction/valid_processed_instruction_data_25k.jsonl"

valid_solutions = 0
total_samples = 0
valid_samples = []

with open(input_file_path, 'r') as infile:
    for line in infile:
        total_samples += 1
        sample = json.loads(line)
        solution_code = sample.get("Solution", "")
        problem_text = sample.get("Problem", "")
        
        # 检查字符串长度
        if len(solution_code) > 2000 or len(problem_text) > 2000:
            print(f"Sample {sample['index']} is filtered out due to length restriction.")
            continue
        
        try:
            # 检查代码的语法
            ast.parse(solution_code)
            valid_solutions += 1
            valid_samples.append(sample)
        except SyntaxError:
            print(f"Sample {sample['index']} has a syntax error in Solution.")

with open(output_file_path, 'w') as outfile:
    for valid_sample in valid_samples:
        outfile.write(json.dumps(valid_sample) + "\n")

print(f"Valid solutions: {valid_solutions}/{total_samples} ({(valid_solutions / total_samples) * 100:.2f}%)")
print(f"Valid samples saved to {output_file_path}")
