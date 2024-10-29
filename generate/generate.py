import torch
import pdb
import argparse

from safetensors.torch import load_file
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import List, Dict
from tqdm import tqdm


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

    for length in lengths:
        min_perplexity = float('inf')
        best_subsequence = None
        
        # Adjust the maximum index for extraction based on actual length
        max_index = actual_length if actual_length < length else actual_length - length + 1

        for i in range(max_index):
            sub_tokens = tokens[i:i + length].unsqueeze(0)  # Shape: [1, length]
            
            with torch.no_grad():
                outputs = model(sub_tokens)
                logits = outputs.logits.squeeze(0)  # Shape: [length, vocab_size]
                
            perplexity = calculate_perplexity(logits, sub_tokens[0])
            
            if perplexity < min_perplexity:
                min_perplexity = perplexity
                best_subsequence = tokenizer.decode(sub_tokens[0], skip_special_tokens=True)

        # If no valid subsequence is found, we might want to store the available tokens up to the max length
        if best_subsequence is None and actual_length > 0:
            # Take the available tokens up to the actual length
            best_subsequence = tokenizer.decode(tokens[:length if length <= actual_length else actual_length], skip_special_tokens=True)
            min_perplexity = float('inf')  # or set to some other default value if you wish

        subsequences[length] = {'subsequence': best_subsequence, 'perplexity': min_perplexity}

    return subsequences


def GEN_SOLUTION(base_checkpoint, model_checkpoint, dp_epsilon=None, step=None, prompt='', max_tokens=100, n=1, temperature=0.8):
    """
    Generate text using the specified model with the provided parameters.
    
    :param base_checkpoint: The path or name of the model checkpoint.
    :param dp_epsilon: Differential privacy parameter (optional, not used in this version).
    :param step: step number for fine-tuned models (optional).
    :param prompt: The input text prompt to generate from.
    :param max_tokens: The maximum number of new tokens to generate.
    :param n: The number of different completions to generate.
    :param temperature: The sampling temperature, controlling randomness.
    :return: List of generated completions.
    """

    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        # use_auth_token=True,
    )

    model = PeftModel.from_pretrained(
        base_model, 
        model_checkpoint
    )

    model = model.merge_and_unload().to(device)

    # model = base_model.to(device)

    # model = AutoModelForCausalLM.from_pretrained(
    #     f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/synthetic_numpy/deepseek-coder-1.3b-instruct/dp{dp_epsilon}_nolora_1norm/checkpoint-{epoch}', 
    #     resume_download=True,
    #     # cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
    #     use_auth_token=True,
    # )


    # inputs = tokenizer.encode('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t"""\n', return_tensors="pt").to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_tokens,
        # eos_token_id=tokenizer.eos_token_id,
        # top_p=0.95,
        num_return_sequences=n,
        temperature=temperature,
        do_sample=True, # 允许随机采样以生成多样化的输出
    )
    # prompt_length = inputs['input_ids'].shape[-1]

    # # Decode only the generated part (excluding the prompt)
    # completion = tokenizer.decode(outputs[0][prompt_length:], clean_up_tokenization_spaces=False)

    results = []

    for i in range(n):
        completion = tokenizer.decode(outputs[i][1:], clean_up_tokenization_spaces=False)
        # extrace top [10, 20, 50] pp tokens 👉 subsequences
        subsequences = extract_low_perplexity_subsequences(completion, tokenizer, model)
        results.append({
            "generated_text": completion,
            "subsequences": subsequences
        })
    # pdb.set_trace()
    # print("generation is: \n", completion)

    # pdb.set_trace()

    return results


def batch_pii_GEN_SOLUTION(
    base_checkpoint, 
    batch_size, 
    dp_epsilon=None,
    step=None,
    prompts='', 
    max_tokens=100, 
    n=1, 
    temperature=0.8
):
    """
    Batch generate solutions with multiple attempts per prompt.
    
    :param base_checkpoint: Model checkpoint path or identifier.
    :param batch_size: Number of prompts per batch.
    :param dp_epsilon: Optional, differential privacy parameter.
    :param step: Optional, model checkpoint step for loading the model.
    :param prompts: List of text prompts.
    :param max_tokens: Maximum tokens for generated responses.
    :param n: Number of generation attempts per prompt.
    :param temperature: Sampling temperature for generation.
    :return: List of generated solutions.
    """
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        # use_auth_token=True,
    )

    # model = PeftModel.from_pretrained(
    #     base_model, 
    #     # f"examples/starcoder/finetune/checkpoints/deepseek-coder-1.3b-instruct/dp{dp_epsilon}/final_checkpoint"
    #     # f"examples/starcoder/finetune/checkpoints/deepseek-coder-6.7b-instruct/dp{dp_epsilon}/final_checkpoint"
    #     f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/synthetic_numpy/deepseek-coder-1.3b-instruct/dp{dp_epsilon}_nolora_1norm/checkpoint-{epoch}'
    # )

    # model = model.merge_and_unload().to(device)

    model = base_model.to(device)

    # model = AutoModelForCausalLM.from_pretrained(
    #     f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/synthetic_numpy/deepseek-coder-1.3b-instruct/dp{dp_epsilon}_nolora_1norm/checkpoint-{epoch}', 
    #     resume_download=True,
    #     # cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
    #     use_auth_token=True,
    # )


    formatted_prompts = [prompts[i: i + batch_size] for i in range(0, len(prompts), batch_size)]
    all_solutions = []

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    generation_config = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }

    for batch in tqdm(formatted_prompts, desc="Generating solutions"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, pad_to_multiple_of=8).to(device)

        # Generate `n` different outputs for each prompt
        for _ in range(n):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )

            # Decode results and extend all_solutions
            solutions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_solutions.extend(solutions)

    return all_solutions


def batch_GEN_SOLUTION(
    base_checkpoint: str, 
    batch_size, 
    prompts: List[str],
    checkpoint_path: str,
    is_pretrained: bool,
    # is_post_step: bool,
    is_baseline: bool
) -> List[str]:

    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    if is_baseline:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            # resume_download=True,
            use_auth_token=True,
        ).to(device)
    else:
        if not is_pretrained:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_checkpoint,
                cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
                use_auth_token=True,
            ) 
            model = PeftModel.from_pretrained(
                base_model, 
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/magicoder/{model_name}/dp{dp_epsilon}/checkpoint-{step}',
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dp{dp_epsilon}_step800_test/checkpoint-{step}',
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/original_data_test/checkpoint-{step}',
                checkpoint_path
            )
            model = model.merge_and_unload().to(device)
            
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_checkpoint,
                cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
                use_auth_token=True,
            ) 
            model = base_model.to(device)


    formatted_prompts = [prompts[i: i + batch_size] for i in range(0, len(prompts), batch_size)]
    all_solutions = []
    
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    generation_config = {
        "do_sample": False,
        "max_new_tokens": 512,
        "num_beams": 1
    }

    for batch in tqdm(formatted_prompts, desc="Generating solutions"):
        # Tokenize each batch
        # pdb.set_trace()
        inputs = tokenizer(batch, return_tensors="pt", padding=True, pad_to_multiple_of=8).to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_config
            )

        # Decode results
        solutions = tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
        # pdb.set_trace()
        all_solutions.extend(solutions)
    
    return all_solutions



if __name__ == '__main__':
    prompt = '\"Password\": '
    # prompt = "Problem:\nI have the following DataFrame:\n    Col1  Col2  Col3  Type\n0      1     2     3     1\n1      4     5     6     1\n2      7     8     9     2\n3    10    11    12     2\n4    13    14    15     3\n5    16    17    18     3\n\n\nThe DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.\nI would like to shuffle the order of the DataFrame's rows according to a list. \nFor example, give a list [2, 4, 0, 3, 1, 5] and desired DataFrame should be:\n    Col1  Col2  Col3  Type\n2      7     8     9     2\n4     13    14    15     3\n0     1     2     3     1\n3    10    11    12     2\n1     4     5     6     1\n5    16    17    18     3\n...\nI want to know how many rows have different Type than the original DataFrame. In this case, 4 rows (0,1,2,4) have different Type than origin.\nHow can I achieve this?\n\n\nA:\n<code>\nimport pandas as pd\nimport numpy as np\n\n\ndf = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],\n                   'Col2': [2, 5, 8, 11, 14, 17],\n                   'Col3': [3, 6, 9, 12, 15, 18],\n                   'Type': [1, 1, 2, 2, 3, 3]})\nList = np.random.permutation(len(df))\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n"
    completion = GEN_SOLUTION(
        base_checkpoint="deepseek-ai/deepseek-coder-6.7b-base", 
        model_checkpoint='/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dp10_syndata/checkpoint-130',
        dp_epsilon='1e10', 
        step=19, 
        prompt=prompt
    )
