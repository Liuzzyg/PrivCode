import json
import random
import os

# Paths to input and output files
CANARY_SAMPLES_PATH = "canary/canary_samples.jsonl"
DATASET_PATH = "data/pii_dataset/raw_dataset/pii_instruction_dataset_python.jsonl"
OUTPUT_PATH_TEMPLATE = "canary/pii_instruction_dataset_canary_rep{}.jsonl"

# Repetition rates for canary injection
REPETITION_RATES = [1, 10, 100]

def read_jsonl(file_path):
    """Read a JSONL file and return a list of JSON objects."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        # Try to fix if it's a JSON array
        return try_fix_json(file_path)
    return data

def try_fix_json(file_path):
    """Attempt to fix a JSON file that is a single array by converting to JSONL."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                # Parse as a JSON array
                array = json.loads(content)
                if isinstance(array, list):
                    # Write back as JSONL
                    with open(file_path, 'w', encoding='utf-8') as f:
                        for item in array:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    print(f"Fixed {file_path} to JSONL format.")
                    return array
    except json.JSONDecodeError as e:
        print(f"Error: Could not fix JSON in {file_path}: {e}")
    return []

def write_jsonl(file_path, data):
    """Write a list of JSON objects to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def inject_canaries(canary_samples, dataset, repetition_rate):
    """Inject canary samples into the dataset with the specified repetition rate."""
    if len(canary_samples) != 5:
        print(f"Error: Expected 5 canary samples, got {len(canary_samples)}.")
        return dataset

    # Create a new dataset starting with the original dataset
    new_dataset = dataset.copy()

    # Inject all canary samples with the given repetition rate
    print(f"Injecting all canary samples with repetition rate {repetition_rate}.")
    for sample in canary_samples:
        new_dataset.extend([sample] * repetition_rate)

    # Shuffle the dataset to randomize the position of canaries
    random.seed(42)  # For reproducibility
    random.shuffle(new_dataset)
    return new_dataset

def main():
    # Read canary samples and dataset
    canary_samples = read_jsonl(CANARY_SAMPLES_PATH)
    dataset = read_jsonl(DATASET_PATH)

    if not canary_samples:
        print("No canary samples loaded. Exiting.")
        return
    if not dataset:
        print("No dataset loaded. Exiting.")
        return

    # Create a separate output file for each repetition rate
    for rate in REPETITION_RATES:
        # Inject canaries for this repetition rate
        new_dataset = inject_canaries(canary_samples, dataset, rate)

        # Save to a file named with the repetition rate
        output_path = OUTPUT_PATH_TEMPLATE.format(rate)
        write_jsonl(output_path, new_dataset)
        print(f"Dataset with repetition rate {rate} saved to {output_path}. Total samples: {len(new_dataset)}")

if __name__ == "__main__":
    main()