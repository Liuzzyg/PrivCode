import json
import random
import pdb
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for progress bars

# Load JSONL file
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

# Save JSONL file
def save_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Extract numpy-related APIs while preserving the original format
def extract_numpy_apis(data):
    numpy_apis = ['numpy.']  # We filter all APIs that start with 'numpy.'
    numpy_related_data = []

    for entry in data:
        new_entry = entry.copy()  # Make a copy of the entry to avoid mutating the original
        apis = entry.get('api_calls', {})
        filtered_apis = {api: calls for api, calls in apis.items() if any(numpy_api in api for numpy_api in numpy_apis)}
        
        if filtered_apis:
            new_entry['api_calls'] = filtered_apis
            numpy_related_data.append(new_entry)
    
    return numpy_related_data

# Count API usage and get frequencies
def count_api_usage(numpy_related_data):
    api_count = defaultdict(int)
    for entry in numpy_related_data:
        unique_apis = set(api.split('(')[0] for api in entry['api_calls'])
        for api in unique_apis:
            api_count[api] += 1
    return api_count

# Calculate sample size for each API
def calculate_sample_sizes(api_count, total_sample_size):
    total_count = sum(api_count.values())
    sample_sizes = {}
    for api, count in api_count.items():
        sample_sizes[api] = max(1, int(count / total_count * total_sample_size))
    return sample_sizes

def get_subset(api_related_data, api_count, subset_ratio):
    # Calculate total number of entries to select based on subset ratio
    total_lines = int(len(api_related_data) * subset_ratio)

    # Calculate sample sizes for each API
    sample_sizes = calculate_sample_sizes(api_count, total_lines)
    
    # Classify entries based on the most frequent API in each entry
    api_entries = defaultdict(list)
    for entry in api_related_data:
        api_frequency = defaultdict(int)
        for api in entry['api_calls']:
            api_name = api.split('(')[0]
            api_frequency[api_name] += 1
        
        # Determine the most frequent API in this entry
        most_frequent_api = max(api_frequency, key=lambda x: (api_frequency[x], x))
        api_entries[most_frequent_api].append(entry)
    
    # Sample entries based on API frequencies
    sampled_entries = []
    for api, entries in api_entries.items():
        num_samples = min(len(entries), sample_sizes.get(api, 0))
        sampled_entries.extend(random.sample(entries, num_samples))
    
    # Ensure the final sample size does not exceed the requested size
    if len(sampled_entries) > total_lines:
        sampled_entries = random.sample(sampled_entries, total_lines)
    
    return sampled_entries

# Main process
def main(input_file, output_file, subset_ratio):
    data = load_jsonl(input_file)
    pdb.set_trace()
    # Extract numpy related data while keeping the original format intact
    numpy_related_data = extract_numpy_apis(data)
    
    # Count API usage frequencies
    api_count = count_api_usage(numpy_related_data)
    
    # Get a subset based on API usage frequencies
    subset = get_subset(numpy_related_data, api_count, subset_ratio)
    
    # Sort entries by their index if available (assuming index is a field in the entries)
    subset = sorted(subset, key=lambda x: x.get('index', 0))
    
    # Save the subset with original structure intact
    save_jsonl(output_file, subset)
    
    print(f"Subset created and saved to {output_file}.")

# Run the process
if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    
    # input_file = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy.jsonl'
    # output_file = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_subset25k.jsonl'
    output_file = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_test.jsonl'
    input_file = '/bigtemp/fzv6en/liuzheng/dpcode/starcoderdata_numpy_subset25k.jsonl'
    
    # subset_ratio = 0.04
    subset_ratio = 0.001
    
    with tqdm(total=100, desc="Processing") as pbar:
        main(input_file, output_file, subset_ratio)
        pbar.update(100)
