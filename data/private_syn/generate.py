import json
import argparse
import pdb
import os
import torch

from tqdm import tqdm
from typing import Any, Dict, List
from datasets import load_dataset  # Import the datasets library for loading the magicoder dataset
from vllm import LLM, SamplingParams

# Function to load jsonl data (if needed for other purposes)
def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Function to save jsonl data
def save_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Define a class for LLMPredictor
class LLMPredictor:
    def __init__(self):
        # Create the LLM object with the desired configuration
        num_GPUs = torch.cuda.device_count()
        print(f"parallel num is {num_GPUs}")
        self.llm = LLM(
            model=ckpt,
            # gpu_memory_utilization=0.5,  # Adjust based on your GPU capacity
            # use_auth_token=True,
            tensor_parallel_size=num_GPUs  # Set to number of GPUs you want to use
        )
        # pdb.set_trace()
        self.sampling_params = SamplingParams(max_tokens=2048)

    def generate_for_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        # Prepare prompts and generate outputs
        prompts = [
            (f"Question: {sample['problem']}\n\nAnswer:") for sample in batch
        ]

        # Generate text for the prompts
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Extract and format the results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text  # Get generated text
            # pdb.set_trace()
            results.append({
                # 'index': batch[i]['index'],
                'problem': batch[i]['problem'],
                'generated_solution': generated_text
            })
        
        return results

# Main function to process input data with progress bar
def main(args, model):
    # Load the magicoder dataset
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        # use_auth_token=True,
        num_proc=args.num_workers,
        cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
    )
    
    # Split into train and test sets
    dataset = dataset.train_test_split(test_size=0.74, seed=args.seed)
    # dataset = dataset.train_test_split(test_size=0.001, seed=args.seed)
    test_data = dataset['test']

    print('Successfully loaded magicoder test data!')

    # Initialize the predictor
    predictor = LLMPredictor()

    # Initialize the progress bar with tqdm
    total_samples = len(test_data)
    pbar = tqdm(total=total_samples, desc="Generating solutions")

    # Initialize an empty list to store results
    all_results = []

    # Process data in batches
    for i in range(0, total_samples, args.batch_size):
        batch = test_data.select(range(i, min(i + args.batch_size, total_samples)))  # Select a batch from test data
        batch_samples = [{'index': idx, 'problem': sample['problem']} for idx, sample in enumerate(batch)]
        results = predictor.generate_for_batch(batch_samples)
        all_results.extend(results)

        # Update the progress bar
        pbar.update(len(batch_samples))

    # Save the generated solutions
    save_jsonl(f'data/private_syn/{model}_private_syndata_55k_dp{args.target_epsilon}.jsonl', all_results)

    # Close the progress bar
    pbar.close()

# Run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--dataset_name", type=str, default="ise-uiuc/Magicoder-OSS-Instruct-75K")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input_column_name", type=str, default="problem")
    parser.add_argument("--output_column_name", type=str, default="solution")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument('--step', type=int, default=50)
    parser.add_argument("--target_epsilon", type=float, default=10)

    args = parser.parse_args()

    model = args.model_path.split("/")[-1]
    base_path = '/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/magicoder'

    ckpt = os.path.join(base_path, f'{model}/dp{args.target_epsilon}/checkpoint-{args.step}')
    print(ckpt)

    main(args, model)
