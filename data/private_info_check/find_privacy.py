import json
from datasets import load_dataset
import pdb

def find_address_in_dataset(dataset_name, subset, split, num_workers, output_file, privacy_info):
    # Load the dataset
    # dataset = load_dataset(
    #     dataset_name,
    #     data_dir=subset,
    #     split=split,
    #     num_proc=num_workers,
    #     cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
    # )
    dataset = load_dataset(
        "json", 
        data_files=dataset_name,
        split=split,
    )
    
    # List to store records containing privacy_info
    records_with_address = []

    # Iterate through the dataset
    for record in dataset:
        problem = record.get('problem', '').lower()
        # solution = record.get('solution', '').lower()
        solution = record.get('generated_solution', '').lower()
        # pdb.set_trace()

        # Check if 'privacy_info' is in the problem or solution
        if privacy_info in problem or privacy_info in solution:
            # Only store 'problem' and 'solution'
            filtered_record = {
                'problem': record.get('problem', ''),
                'solution': record.get('generated_solution', '')
            }
            records_with_address.append(filtered_record)
    
    if len(records_with_address) != 0:
        # Write results to a JSONL file
        with open(output_file, 'w') as f:
            for record in records_with_address:
                json.dump(record, f)
                f.write('\n')  # JSONL requires each JSON object to be on a new line

    print(f"Found {len(records_with_address)} records containing {privacy_info}. Results saved to {output_file}.")


# Usage Example:
privacy_info = 'github.com'

find_address_in_dataset(
    # dataset_name="ise-uiuc/Magicoder-OSS-Instruct-75K",
    dataset_name='data/private_syn/private_syndata_55k_dp10.jsonl',
    subset=None,
    split="train",  # or 'validation', 'test' based on your dataset split
    num_workers=None,  # adjust according to your system
    output_file=f"data/private_syn/privacy_info_{privacy_info}.jsonl",
    privacy_info=privacy_info
)
