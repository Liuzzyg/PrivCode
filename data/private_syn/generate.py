import json
import argparse
import pdb
import os
import torch

from tqdm import tqdm
from typing import Any, Dict, List
from datasets import load_dataset
from vllm import LLM, SamplingParams

def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(output_path, data):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

class LLMPredictor:
    def __init__(self):
        num_GPUs = torch.cuda.device_count()
        print(f"parallel num is {num_GPUs}")
        self.llm = LLM(
            model=ckpt,
            # gpu_memory_utilization=0.5,  # Adjust based on your GPU capacity
            # use_auth_token=True,
            tensor_parallel_size=num_GPUs
        )
        # pdb.set_trace()
        self.sampling_params = SamplingParams(max_tokens=2048)

    def generate_for_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        prompts = [
            (f"Question: {sample['problem']}\n\nAnswer:") for sample in batch
        ]

        outputs = self.llm.generate(prompts, self.sampling_params)

        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            results.append({
                'problem': batch[i]['problem'],
                'generated_solution': generated_text
            })
        
        return results

def main(args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        num_proc=args.num_workers,
        cache_dir='.../.cache/huggingface/datasets'
    )
    
    dataset = dataset.train_test_split(test_size=0.74, seed=args.seed)
    test_data = dataset['test']
    print('Successfully loaded magicoder test data!')

    predictor = LLMPredictor()

    total_samples = len(test_data)
    pbar = tqdm(total=total_samples, desc="Generating solutions")

    all_results = []

    for i in range(0, total_samples, args.batch_size):
        batch = test_data.select(range(i, min(i + args.batch_size, total_samples)))
        batch_samples = [{'index': idx, 'problem': sample['problem']} for idx, sample in enumerate(batch)]
        results = predictor.generate_for_batch(batch_samples)
        all_results.extend(results)
        pbar.update(len(batch_samples))
    save_jsonl(args.save_path, all_results)
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="ise-uiuc/Magicoder-OSS-Instruct-75K")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--input_column_name", type=str, default="problem")
    parser.add_argument("--output_column_name", type=str, default="solution")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=None)

    args = parser.parse_args()

    ckpt = args.ckpt
    print("load checkpoint from: " + ckpt)

    main(args)
