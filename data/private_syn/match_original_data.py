import json
import argparse

from datasets import load_dataset, load_from_disk, Features
from tqdm import tqdm

# Function to create a dictionary from problems to data for fast lookup
def create_problem_mapping(data):
    problem_map = {}
    for sample in data:
        problem = sample.get('problem', '').strip().lower()
        solution = sample.get('solution', '')
        lang = sample.get('lang', 'unknown')
        problem_map[problem] = {'solution': solution, 'lang': lang}
    return problem_map

# Function to match and save the data to a new JSONL file
def match_and_save_data(original_data, local_data_file, output_file):
    # Create a dictionary to store the matched results
    matched_data = []

    # Load the local cleaned data
    private_data = load_dataset(
        "json", 
        data_files=local_data_file,
        split='train',
        # features=features
    )

    # Create a hash map for fast problem lookup from magicoder data
    problem_map = create_problem_mapping(original_data)
    print('Created problem mapping for magicoder dataset!')

    # Iterate over the local data and match with magicoder dataset
    for entry in tqdm(private_data, desc="Matching data"):
        problem = entry.get('problem', '').strip().lower()

        # If the problem exists in the magicoder dataset
        if problem in problem_map:
            matched_entry = {
                'lang': problem_map[problem]['lang'],
                'problem': problem,
                'solution': problem_map[problem]['solution'],
            }
            matched_data.append(matched_entry)

    print('original data length is', len(matched_data))
    # Write the matched data to a new JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in matched_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Matched data saved to {output_file}")

# Example usage
def main(args):
    # Load Magicoder-OSS-Instruct-75K dataset from Hugging Face
    original_data = load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train')

    print('Successfully loaded original dataset!')

    model = args.model.split("/")[-1]

    # File paths
    local_data_file = f'data/private_syn/{model}_cleaned_private_syndata.jsonl'  # Path to your cleaned local data
    output_file = f'data/private_syn/{model}_original_data.jsonl'  # Path to save the matched raw data

    # Perform matching and saving
    match_and_save_data(original_data, local_data_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")

    args = parser.parse_args()

    main(args)