import json
import pdb
from tqdm import tqdm
from typing import Any, Dict, List
from vllm import LLM, SamplingParams
import argparse

# Function to load jsonl data
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
        self.llm = LLM(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            # gpu_memory_utilization=0.5,  # Adjust based on your GPU capacity
            trust_remote_code=True,
            download_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
            tensor_parallel_size=4  # Set to number of GPUs you want to use
        )
        self.sampling_params = SamplingParams(max_tokens=1024)

    def generate_for_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        # Prepare prompts and generate outputs
        prompts = [
            (
                "Please gain inspiration from the following random code snippet related to the Numpy API to create a high-quality and simple programming problem. We need that the created programming problem only uses numpy api, and don't use any other libraries. Present your output in two distinct sections: [Problem Description]: and [Solution]:, and each section end up with [END].\n\n"
                f"Code snippet for inspiration:{sample['content']}\n\n"
                "Guidelines for each section:\n\n"
                "1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included.\n\n"
                "2. [Solution]: Offer a comprehensive, **correct** solution that accurately addresses the [Problem Description] you provided, and use **only the NumPy api**."
                "You only need to give your output with above two sections, with the format:[Problem Description]:\n[content][END]\n[Solution]:\n[content][END]"
            )
            for sample in batch
        ]

        # Generate text for the prompts
        outputs = self.llm.generate(prompts, self.sampling_params)

        # pdb.set_trace()
        # Extract and format the results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text  # Get generated text
            results.append({
                'index': batch[i]['index'],
                'completion': generated_text
            })
        
        return results

# Main function to process input data with progress bar
def main(input_path, output_path, batch_size=64):
    # Load input data
    data = load_jsonl(input_path)
    print('Successfully loaded input data!')

    # Initialize the predictor
    predictor = LLMPredictor()

    # Initialize the progress bar with tqdm
    total_samples = len(data)
    pbar = tqdm(total=total_samples, desc="Generating problems and solutions")

    # Initialize an empty list to store results
    all_results = []

    # Process data in batches
    for i in range(0, total_samples, batch_size):
        batch = data[i:i + batch_size]
        results = predictor.generate_for_batch(batch)
        all_results.extend(results)

        # Update the progress bar
        pbar.update(len(batch))

    # Save the generated completions
    save_jsonl(output_path, all_results)

    # Close the progress bar
    pbar.close()

# Input and output paths
input_path = 'data/oss_instruction/random_numpy_code_snippet_25k.jsonl'
output_path = 'data/oss_instruction/synthetic_numpy_instruction_data_25k.jsonl'
# input_path = 'data/oss_instruction/random_numpy_code_snippet_test.jsonl'
# output_path = 'data/oss_instruction/synthetic_numpy_instruction_data_25.jsonl'

# Run the main function
if __name__ == "__main__":
    main(input_path, output_path)
