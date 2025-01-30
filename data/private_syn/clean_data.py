import json
import ast
import pdb
import argparse

from datasets import load_dataset
from tqdm import tqdm

# Function to create a dictionary from problems to languages for fast lookup
def create_problem_language_mapping(original_data):
    problem_language_map = {}
    for original_sample in original_data:
        problem = original_sample.get('problem', '').strip().lower()
        language = original_sample.get('lang', 'unknown')
        problem_language_map[problem] = language
    return problem_language_map

# Function to check if a string is valid Python code using ast
def is_valid_python(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

# Function to clean and filter the data, and add language
def clean_data_with_language(input_file, output_file):
    cleaned_data = []
    language_classes = set()  # Set to keep track of unique languages

    # Load dataset
    dataset = load_dataset(
        'ise-uiuc/Magicoder-OSS-Instruct-75K',
        split='train',
    )

    # Split into train and test sets
    dataset = dataset.train_test_split(test_size=0.74, seed=42)
    test_data = dataset['test']

    print('Successfully loaded magicoder test data!')

    # Create a hash map for fast problem-language matching
    problem_language_map = create_problem_language_mapping(test_data)
    print('Created problem-language mapping!')

    # Get total number of lines in input file for progress bar
    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    # Read the input JSONL file with progress bar
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in tqdm(infile, total=total_lines, desc="Processing data"):
            # Parse each line as a JSON object
            data = json.loads(line)
            problem = data.get('problem', '')
            solution = data.get('generated_solution', '')

            # Find the first and second occurrence of triple backticks
            code_sections = solution.split('```')

            # If there are at least two occurrences of triple backticks
            if len(code_sections) > 2:
                cleaned_solution = code_sections[1].strip()
                
                if len(cleaned_solution) > 1400:
                    continue
                
                # pdb.set_trace()
            
                # Use the hash map to find the matched language
                matched_language = problem_language_map.get(problem.strip().lower(), 'unknown')

                # Add the matched language to the set
                language_classes.add(matched_language)

                # Only perform Python syntax check for Python solutions
                if matched_language == 'python':
                    if is_valid_python(cleaned_solution):
                        cleaned_data.append({
                            'language': matched_language,
                            'problem': problem,
                            'solution': "'''" + cleaned_solution + " '''",
                        })
                else:
                    # For other languages, simply append the cleaned data
                    cleaned_data.append({
                        'language': matched_language,
                        'problem': problem,
                        'solution': "'''" + cleaned_solution + " '''",
                    })

    # Write the cleaned data to a new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in cleaned_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Output the unique language classes
    print("Unique languages matched in the data:")
    print(language_classes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct")

    args = parser.parse_args()

    model = args.model.split("/")[-1]
    input_file = f'data/private_syn/{model}/private_syndata_55k_dp10.jsonl'
    output_file = f'data/private_syn/{model}/cleaned_private_syndata.jsonl'
    clean_data_with_language(input_file, output_file)

    print(f"Data cleaning and filtering completed. Cleaned data with language saved to {output_file}.")
