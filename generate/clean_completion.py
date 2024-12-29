import os
import json

def clean_jsonl_file(file_path):
    # Rename the original file to add '_uncleaned' suffix
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)
    uncleaned_file_path = os.path.join(dir_name, f"{base_name}_uncleaned{ext}")
    os.rename(file_path, uncleaned_file_path)

    # Open the uncleaned file for reading and a new file for writing
    with open(uncleaned_file_path, 'r', encoding='utf-8') as uncleaned_file, \
         open(file_path, 'w', encoding='utf-8') as cleaned_file:

        for line in uncleaned_file:
            try:
                # Parse the line as JSON
                record = json.loads(line)

                # Check and clean the 'solution' field if it ends with triple quotes
                if 'solution' in record and record['solution'].endswith("'''"):
                    record['solution'] = record['solution'][:-3].rstrip()

                # Write the cleaned record back to the new file
                cleaned_file.write(json.dumps(record, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")

# Example usage
clean_jsonl_file('generate/evalplus_0.3.1/deepseek-coder-6.7b-base/step2_codeonly/mbpp/bigtemp--fzv6en--liuzheng--dpcode--checkpoints_codeonly--step2_promptsim_Llama-3.1-70B-Instruct_tau0.82--deepseek-coder-6.7b-base_dp4_lambda1000to0.1_alpha0.01_datasize55500--privsyn_merged--checkpoint-100_vllm_temp_0.0.raw.jsonl')
