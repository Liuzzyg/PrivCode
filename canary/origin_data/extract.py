import json
import os

input_file = "data/pii_dataset/raw_dataset/pii_instruction_dataset_python.jsonl"
output_dir = ".../canary"
output_file = os.path.join(output_dir, "extracted_samples.jsonl")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read and extract 20 samples
samples = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        samples.append(json.loads(line.strip()))

# Save extracted samples
with open(output_file, 'w', encoding='utf-8') as f:
    for sample in samples:
        f.write(json.dumps(sample) + '\n')

print(f"Extracted 20 samples and saved to {output_file}")