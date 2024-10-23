import json
import time
import pdb
import argparse
import torch

from typing import Any, Dict, List
from vllm import LLM, SamplingParams
from tqdm import tqdm
from Prompt import get_all_queries
from transformers import AutoModelForCausalLM, AutoTokenizer


# Function to load jsonl data (if needed for other purposes)
def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Function to save jsonl data
def save_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def calculate_perplexity(logits, target_ids):
    probabilities = torch.softmax(logits, dim=-1)
    
    target_probs = probabilities[range(len(target_ids)), target_ids]
    
    perplexity = torch.exp(-torch.log(target_probs).mean())
    return perplexity.item()

def extract_low_perplexity_subsequences(text, tokenizer, model, lengths=[10, 20, 50]):
    tokens = tokenizer.encode(text, return_tensors="pt").squeeze().to(model.device)
    subsequences = {}

    # Get the actual length of the tokens
    actual_length = tokens.size(0)
    pdb.set_trace()

    for length in lengths:
        min_perplexity = float('inf')
        best_subsequence = None
        
        # Set max_index such that sub_tokens[i:i + length] won't exceed actual_length
        max_index = actual_length - length + 1 if actual_length >= length else 0

        if actual_length <= length:
            # If the actual length is smaller than or equal to the desired length, use the full sequence
            sub_tokens = tokens.unsqueeze(0)  # Shape: [1, actual_length]
            with torch.no_grad():
                outputs = model(sub_tokens)
                logits = outputs.logits.squeeze(0)  # Shape: [actual_length, vocab_size]
                
            perplexity = calculate_perplexity(logits, tokens)
            best_subsequence = tokenizer.decode(tokens, skip_special_tokens=True)
            min_perplexity = perplexity
        else:
            # Sliding window over the sequence for the current length
            for i in range(max_index):
                sub_tokens = tokens[i:i + length].unsqueeze(0)  # Shape: [1, length]
                
                with torch.no_grad():
                    outputs = model(sub_tokens)
                    logits = outputs.logits.squeeze(0)  # Shape: [length, vocab_size]
                    
                perplexity = calculate_perplexity(logits, sub_tokens[0])
                6
                if perplexity < min_perplexity:
                    min_perplexity = perplexity
                    best_subsequence = tokenizer.decode(sub_tokens[0], skip_special_tokens=True)

        # Store the best subsequence for the current length
        subsequences[length] = {'subsequence': best_subsequence, 'perplexity': min_perplexity}
        pdb.set_trace()

    return subsequences



# Define a class for LLMPredictor
class LLMPredictor:
    def __init__(self):
        # Create the LLM object with the desired configuration
        num_GPUs = torch.cuda.device_count()
        print(f"parallel num is {num_GPUs}")
        self.llm = LLM(
            model=args.model_checkpoint,
            gpu_memory_utilization=0.5,  # Adjust based on your GPU capacity
            # use_auth_token=True,
            tensor_parallel_size=num_GPUs  # Set to number of GPUs you want to use
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_checkpoint,
            cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
            # use_auth_token=True,
        )

    def generate(self, dp_epsilon=None, step=None, prompt='', max_tokens=100, n=1, temperature=0.8):
        results = []
        # Prepare prompts and generate outputs
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=n
        )
        # pdb.set_trace()
        # Generate text for the prompts
        outputs = self.llm.generate(prompt, self.sampling_params)

        # for i in range(n):
        #     completion = outputs[0].outputs[i].text
        #     # extrace top [10, 20, 50] pp tokens 👉 subsequences
        #     subsequences = extract_low_perplexity_subsequences(completion, self.tokenizer, self.model)
        #     results.append({
        #         "generated_text": completion,
        #         "subsequences": subsequences
        #     })
        
        for i in range(n):
            completion = outputs[0].outputs[i].text
            results.append({
                "generated_text": completion,
            })

        return results


def main(args):
    # Load the queries prompts
    queries = get_all_queries()
    pdb.set_trace()

    # Initialize the predictor
    predictor = LLMPredictor()

    # Initialize an empty list to store results
    all_results = []

    for query in tqdm(queries):
        prompt = query.prompt.text
        # pdb.set_trace()
        
        results = predictor.generate(
            prompt=prompt, 
            n=5, 
            temperature=query.params['temperature']
        )

        for result in results:
            # ordered_result = {
            #     'prompt': prompt,
            #     'generated_text': result['generated_text'],
            #     'subsequences': result['subsequences']
            # }
            
            ordered_result = {
                'prompt': prompt,
                'generated_text': result['generated_text']
            }
            all_results.append(ordered_result)

    # Save the generated solutions
    # pdb.set_trace()
    save_jsonl(args.output_file, all_results)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_checkpoint", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct")
    parser.add_argument("--model_checkpoint", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dp10_syndata_merged/checkpoint-360")
    parser.add_argument("--output_file", type=str, default='pii_leakage/queries/deepseek-coder-6.7b-base_dp10_syndata_step360_vllm.jsonl')

    # parser.add_argument("--model_checkpoint", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/original_data_merged/checkpoint-360")
    # parser.add_argument("--output_file", type=str, default='pii_leakage/queries/deepseek-coder-6.7b-base_original_data_step360_vllm.jsonl')
    args = parser.parse_args()

    main(args)