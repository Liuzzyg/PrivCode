import re
import json
import pdb
import os
from tqdm import tqdm
from typing import Any, Dict, List
from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset

# Function to load jsonl data
def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Function to save jsonl data
def save_jsonl(output_path, data):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Define a class for LLMPredictor
class LLMPredictor:
    def __init__(self):
        # Create the LLM object with the desired configuration
        self.llm = LLM(
            model="meta-llama/Llama-3.1-70B-Instruct",
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            trust_remote_code=True,
            download_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
            tensor_parallel_size=4,
            # gpu_memory_utilization=0.8,
        )
        self.sampling_params = SamplingParams(max_tokens=2048)

    def generate_for_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        # pdb.set_trace()
        # Prepare prompts and generate outputs
        prompts = [
            (
                "Please take inspiration from the following random code snippet to create a high-quality and simple programming problem formatted as a problem-solution pair, only using python language. These code snippets may represent a self-contained program or a part of a larger program. Each code snippet contains some personal privacy information, such as email addresses, IP addresses, keys, names, usernames, or passwords. In the generated solution, make sure to retain this personal privacy information, but it must not appear in the problem description. Present your output in two distinct sections: [Problem] and [Solution], and ensure that each section ends with [END].\n\n"
                f"Code snippet for inspiration:{sample}\n\n"
                "Guidelines for each section:\n\n"
                "1. [Problem Description]: This section should be **completely self-contained**, providing all the contextual information necessary to understand and solve the problem. Assume common programming knowledge, but explicitly include any specific context, variables, or smaller heuristic code snippets relevant to the problem. Ensure **no personal privacy information** from the code snippet appears in this section.\n\n"
                "2. [Solution]: Provide a comprehensive, **correct** solution that accurately addresses the [Problem], formatted as ```python\ncode```[END]. **Retain the personal privacy information** from the raw code snippet in this section.\n\n"
                "You only need to give your output with above two sections, with the format:[Problem]:\n[content][END]\n[Solution]:\n[content][END].\n\n"
                "Here is the problem-solution pair. [Problem]:\n"
            )
            for sample in batch['text']
        ]
        # pdb.set_trace()

        # Generate text for the prompts
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Extract and format the results
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text  # Get generated text
            # pdb.set_trace()

            # Post-process the generated text
            problem, solution = self.post_process(generated_text)

            if problem and solution:
                results.append({
                    'problem': problem,
                    'solution': solution
                })
        
        return results

    @staticmethod
    def post_process(generated_text: str):
        # Define regular expressions for extraction
        problem_pattern = r"(.*?)\[END\]"
        solution_pattern = r"\[Solution\]:\n(.*?)\[END\]"

        problem_match = re.search(problem_pattern, generated_text, re.DOTALL)
        solution_match = re.search(solution_pattern, generated_text, re.DOTALL)

        if not problem_match or not solution_match:
            return None, None

        problem = problem_match.group(1).strip()
        solution = solution_match.group(1).strip()

        # Skip invalid samples
        if not problem or not solution or 'Solution' in problem.lower():
            return None, None

        return problem, solution

# Main function to process input data with progress bar
def main(output_path, batch_size=64):
    # Load input data
    dataset = load_dataset(
        'terryyz/pii',
        split='test',
        cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
    )
    # only train split
    dataset = dataset.train_test_split(train_size=0.99999, seed=42)
    # dataset = dataset.train_test_split(train_size=0.01, seed=42)
    train_data = dataset['train']
    print('Successfully loaded input data!')

    # Initialize the predictor
    predictor = LLMPredictor()

    # Initialize the progress bar with tqdm
    total_samples = len(train_data)
    pbar = tqdm(total=total_samples, desc="Generating problems and solutions")

    # Initialize an empty list to store results
    all_results = []

    # Process data in batches
    for i in range(0, total_samples, batch_size):
        # pdb.set_trace()
        batch = train_data[i:i + batch_size]
        results = predictor.generate_for_batch(batch)
        
        # pdb.set_trace()
        all_results.extend(results)

        # Update the progress bar
        pbar.update(len(batch))

    # Save the generated completions
    save_jsonl(output_path, all_results)

    # Close the progress bar
    pbar.close()

# Input and output paths
output_path = 'data/pii_dataset/pii_instruction_dataset_python.jsonl'

# Run the main function
if __name__ == "__main__":
    main(output_path)
