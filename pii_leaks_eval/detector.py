import argparse
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import tqdm
import torch
import pdb
import os
import numpy as np
from prompt_template import generate_pii_prompt_signature


class CodeLLMDetector:
    def __init__(self, model_path, cache_dir, max_tokens, temperature, n, seed):
        # Load the code LLM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_GPUs = torch.cuda.device_count()
        self.llm = LLM(
            model=model_path,
            download_dir=cache_dir,
            tensor_parallel_size=num_GPUs,
            max_model_len=8000,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            n=n
        )

        # Load the NER model for PII detection
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starpii")
        self.model = AutoModelForTokenClassification.from_pretrained(
            "bigcode/starpii",
            cache_dir=cache_dir,
            # device_map="auto"
        ).to(device)

        self.ner_pipeline = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple", 
            device=device
        ) 

    def generate_and_detect(self, prompt_num, output_path):
        # Generate prompts
        prompts = generate_pii_prompt_signature(prompt_num, seed=self.sampling_params.seed)

        # Progress bar for generation
        generated_solutions = []
        print("Generating solutions...")
        for prompt in tqdm.tqdm(prompts, desc="Prompt generation"):
            solutions = self.llm.generate(prompt, sampling_params=self.sampling_params)
            for i, solution in enumerate(solutions):
                for _, output in enumerate(solution.outputs):
                    generated_solutions.append({"prompt": prompt, "solution": output.text})
                    # pdb.set_trace()

        # Progress bar for detection
        print("Detecting PII entities...")
        entity_count = 0
        results_to_save = []
        
        for generated in tqdm.tqdm(generated_solutions, desc="PII Detection"):
            solution = generated["solution"]
            try:
                # Attempt to detect PII entities in the solution
                results = self.ner_pipeline(solution)
                entity_count += len(results)
                results_to_save.append({"prompt": generated["prompt"], "solution": solution, "entities": results})
            except Exception as e:
                # Skip this sample if an error occurs
                print(f"Error processing solution for prompt: {generated['prompt']}. Error: {e}")
                continue
            
        print(f"Total entities detected: {entity_count}")
        results_to_save.append({"total_entities_detected": entity_count})

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save results to output path
        with open(output_path, "w") as f:
            for result in results_to_save:
                # Ensure all float32 values are converted to regular Python floats
                result = self.convert_float32(result)
                f.write(json.dumps(result) + "\n")

        return entity_count, generated_solutions

    def convert_float32(self, data):
        """Convert float32 values in the data to Python native float."""
        if isinstance(data, dict):
            return {key: self.convert_float32(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.convert_float32(item) for item in data]
        elif isinstance(data, np.float32):
            return float(data)  # Convert np.float32 to native Python float
        else:
            return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and detect PII using Code LLM.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--cache_dir", type=str, default="/bigtemp/fzv6en/.cache/huggingface/hub", help="Cache directory for models.")
    parser.add_argument("--prompt_num", type=int, required=True, help="Number of prompts to generate.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--n", type=int, default=4, help="Number of completions to generate per prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results.")

    args = parser.parse_args()

    detector = CodeLLMDetector(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        n=args.n,
        seed=args.seed
    )

    entity_count, solutions = detector.generate_and_detect(
        prompt_num=args.prompt_num,
        output_path=args.output_path
    )
