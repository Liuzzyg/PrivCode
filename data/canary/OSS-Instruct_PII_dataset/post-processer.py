import json
from tqdm import tqdm


# Function to load jsonl data
def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Function to save jsonl data
def save_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Function to filter data based on solution length
def filter_data(input_path, output_path, max_solution_length=5170):
    # Load the raw dataset
    raw_data = load_jsonl(input_path)

    # Initialize an empty list for filtered data
    filtered_data = []

    # Iterate through each entry in the raw dataset
    for entry in tqdm(raw_data, desc="Filtering data"):
        solution = entry.get('solution', '')

        # Skip entries with solution exceeding max length
        if len(solution) > max_solution_length:
            continue

        filtered_data.append(entry)

    # Save the filtered data to a new JSONL file
    save_jsonl(output_path, filtered_data)

# Paths to input and output files
input_path = 'data/pii_dataset/raw_dataset/pii_instruction_dataset_raw.jsonl'
output_path = 'data/pii_dataset/raw_dataset/pii_instruction_dataset.jsonl'

# Run the filtering process
if __name__ == "__main__":
    filter_data(input_path, output_path)
