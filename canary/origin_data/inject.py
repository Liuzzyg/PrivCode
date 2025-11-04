import json
import random
import os
from datasets import load_dataset

# Paths to input and output files
CANARY_SAMPLES_PATH = "canary/origin_data/canary_samples.jsonl"
DATASET_PATH = "data/canary/OSS-Instruct_PII_dataset/pii_instruction_dataset.jsonl"
DATASET_REPO = "ZhengLiu33/OSS-Instruct-PII-dataset"
OUTPUT_PATH_TEMPLATE = "canary/origin_data/pii_instruction_dataset_canary_rep{}.jsonl"

# Repetition rates for canary injection
REPETITION_RATES = [5, 10, 100]

def read_jsonl(file_path):
    """Read a JSONL file and return a list of JSON objects."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return try_fix_json(file_path)
    return data

def try_fix_json(file_path):
    """Attempt to fix a JSON file that is a single array by converting to JSONL."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                array = json.loads(content)
                if isinstance(array, list):
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
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def inject_canaries(canary_samples, dataset, repetition_rate):
    """Inject canary samples into the dataset with the specified repetition rate."""
    if len(canary_samples) != 5:
        print(f"Error: Expected 5 canary samples, got {len(canary_samples)}.")
        return dataset

    new_dataset = dataset.copy()

    print(f"Injecting all canary samples with repetition rate {repetition_rate}.")
    for sample in canary_samples:
        new_dataset.extend([sample] * repetition_rate)

    random.seed(42)
    random.shuffle(new_dataset)
    return new_dataset

def load_dataset_with_fallback(dataset_path):
    """Try to load dataset locally; if not found, download from Hugging Face."""
    if os.path.exists(dataset_path):
        print(f"✅ Local dataset found at {dataset_path}.")
        return read_jsonl(dataset_path)
    else:
        print(f"⚠️ Local dataset not found. Trying to download from Hugging Face...")
        try:
            hf_dataset = load_dataset(DATASET_REPO, split="train")
            data = hf_dataset.to_list()
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            write_jsonl(dataset_path, data)
            print(f"✅ Downloaded and cached dataset to {dataset_path}")
            return data
        except Exception as e:
            print(f"❌ Failed to download dataset from Hugging Face: {e}")
            return []

def main():
    canary_samples = read_jsonl(CANARY_SAMPLES_PATH)
    dataset = load_dataset_with_fallback(DATASET_PATH)

    if not canary_samples:
        print("No canary samples loaded. Exiting.")
        return
    if not dataset:
        print("No dataset loaded. Exiting.")
        return

    for rate in REPETITION_RATES:
        new_dataset = inject_canaries(canary_samples, dataset, rate)
        output_path = OUTPUT_PATH_TEMPLATE.format(rate)
        write_jsonl(output_path, new_dataset)
        print(f"✅ Dataset with repetition rate {rate} saved to {output_path}. Total samples: {len(new_dataset)}")

if __name__ == "__main__":
    main()
