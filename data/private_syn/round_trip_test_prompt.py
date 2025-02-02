import argparse
import json
import pdb
from collections import defaultdict
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
from bert_score import score as bert_score
import re
import os



def generate_descriptions_batch(llm, generated_data, sampling_params, batch_size=8):
    described_data = []
    prompts = []
    samples_mapping = {}

    for idx, sample in enumerate(generated_data):
        solution = sample.get("solution", "")
        
        if not solution:
            continue
        
        prompt = (
            f"The following code is a solution to a natural language programming problem. "
            f"Please describe the problem into the format **You are tasked with [full described problem] [END]** according to the given solution:\n"
            f"Given solution:\n{solution}\n"
            f"Described problem:\n"
        )
        prompts.append(prompt)
        samples_mapping[len(prompts) - 1] = sample

    with tqdm(total=len(prompts), desc="Generating descriptions", unit="batch") as pbar:
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            responses = llm.generate(batch_prompts, sampling_params)

            pattern = r"\*\*You are tasked with (.+?) \[END\]\*\*"

            for local_idx, response in enumerate(responses):
                global_idx = batch_start + local_idx
                valid_description = None
                for output in response.outputs:
                    match = re.search(pattern, output.text)
                    if match:
                        valid_description = match.group(1).strip()
                        break
                
                if valid_description:
                    sample = samples_mapping[global_idx]
                    sample["described_prompt"] = "You are tasked with " + valid_description
                    described_data.append(sample)
            
            pbar.update(len(batch_prompts))
    
    return described_data


def filter_by_prompt_similarity(described_data, threshold):
    filtered_samples = []
    for sample in tqdm(described_data, desc="Filtering by prompt similarity", unit="sample"):
        original_prompt = sample.get("problem", "")
        described_prompt = sample.get("described_prompt", "")

        # pdb.set_trace()
        P, R, F1 = bert_score(cands=[described_prompt], refs=[original_prompt], lang="en")
        f1_score = F1.item()

        if f1_score >= threshold:
            filtered_samples.append(sample)

    return filtered_samples


def save_filtered_data(filtered_data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for sample in tqdm(filtered_data, desc="Saving filtered data", unit="sample"):
                simplified_sample = {
                    "problem": sample.get("problem", ""),
                    "solution": sample.get("solution", ""),
                }
                outfile.write(json.dumps(simplified_sample, ensure_ascii=False) + "\n")
        print(f"Saved {len(filtered_data)} filtered samples to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise


def main(args):
    with open(args.input_path, "r") as f:
        generated_data = [json.loads(line) for line in f]

    # with open(args.input_path, "r") as f:
    #     generated_data = [json.loads(line) for line in f][:100]  

    num_GPUs = torch.cuda.device_count()
    llm = LLM(
        model=args.round_trip_model, 
        tensor_parallel_size=num_GPUs, 
        gpu_memory_utilization=0.8,
        download_dir=".../.cache/huggingface/hub"
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens, n=args.generated_num, temperature=args.temperature)

    described_data = generate_descriptions_batch(llm, generated_data, sampling_params, batch_size=args.batch_size)

    filtered_data = filter_by_prompt_similarity(described_data, threshold=args.sim_threshold)

    save_filtered_data(filtered_data, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data configs
    parser.add_argument("--input_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--output_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B")

    # round trip configs
    # parser.add_argument("--round_trip_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--round_trip_model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--sim_threshold", type=float, default=0.82)
    parser.add_argument("--batch_size", type=int, default=128)

    # vllm configs
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--generated_num", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()

    main(args)
