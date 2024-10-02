import torch
import pdb

from safetensors.torch import load_file
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from typing import List, Dict
from tqdm import tqdm



def GEN_SOLUTION_deepseek(base_checkpoint, dp_epsilon, epoch, prompt):
    # single sample test

    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_checkpoint,
    #     cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
    #     use_auth_token=True,
    # )

    # model = PeftModel.from_pretrained(
    #     base_model, 
    #     # f"examples/starcoder/finetune/checkpoints/deepseek-coder-1.3b-instruct/dp{dp_epsilon}/final_checkpoint"
    #     # f"examples/starcoder/finetune/checkpoints/deepseek-coder-6.7b-instruct/dp{dp_epsilon}/final_checkpoint"
    #     f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/synthetic_numpy/deepseek-coder-1.3b-instruct/dp{dp_epsilon}_nolora_1norm/checkpoint-{epoch}'
    # )

    # model = model.merge_and_unload().to(device)

    # model = base_model.to(device)

    model = AutoModelForCausalLM.from_pretrained(
        f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/synthetic_numpy/deepseek-coder-1.3b-instruct/dp{dp_epsilon}_nolora_1norm/checkpoint-{epoch}', 
        resume_download=True,
        # cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        use_auth_token=True,
    )


    # inputs = tokenizer.encode('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t"""\n', return_tensors="pt").to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        # top_p=0.95,
        # temperature=0.8,
        # do_sample=True,
    )
    prompt_length = inputs['input_ids'].shape[-1]

    # Decode only the generated part (excluding the prompt)
    completion = tokenizer.decode(outputs[0][prompt_length:], clean_up_tokenization_spaces=False)


    # # clean_up_tokenization_spaces=False prevents a tokenizer edge case which can result in spaces being removed around punctuation
    
    # completion = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)

    print(completion)

    return completion



def batch_GEN_SOLUTION(
    base_checkpoint: str, 
    batch_size, 
    prompts: List[str],
    checkpoint_path: str,
    is_pretrained: bool,
    is_post_step: bool,
) -> List[str]:

    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    if not is_pretrained:
        if is_post_step:
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
            model = AutoModelForCausalLM.from_pretrained(
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/magicoder/{model_name}/nodp/checkpoint-{step}',  
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/magicoder/{model_name}/dp{dp_epsilon}/checkpoint-{step}', 
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/magicoder/{model_name}/dp{dp_epsilon}_lbs2024/checkpoint-{step}', 
                # f'/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/{model_name}/dp{dp_epsilon}_step800/checkpoint-{step}',
                checkpoint_path,
                resume_download=True,
                use_auth_token=True,
            ).to(device)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_checkpoint,
            cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
            use_auth_token=True,
        ) 
        model = base_model.to(device)


    formatted_prompts = [prompts[i: i + batch_size] for i in range(0, len(prompts), batch_size)]
    all_solutions = []

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
    prompt = 'import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nx = np.arange(10)\ny = np.arange(10)\n\n# Plot y over x\n# Show legend and use the greek letter lambda as the legend label\n# SOLUTION START\n'
    # prompt = "Problem:\nI have the following DataFrame:\n    Col1  Col2  Col3  Type\n0      1     2     3     1\n1      4     5     6     1\n2      7     8     9     2\n3    10    11    12     2\n4    13    14    15     3\n5    16    17    18     3\n\n\nThe DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.\nI would like to shuffle the order of the DataFrame's rows according to a list. \nFor example, give a list [2, 4, 0, 3, 1, 5] and desired DataFrame should be:\n    Col1  Col2  Col3  Type\n2      7     8     9     2\n4     13    14    15     3\n0     1     2     3     1\n3    10    11    12     2\n1     4     5     6     1\n5    16    17    18     3\n...\nI want to know how many rows have different Type than the original DataFrame. In this case, 4 rows (0,1,2,4) have different Type than origin.\nHow can I achieve this?\n\n\nA:\n<code>\nimport pandas as pd\nimport numpy as np\n\n\ndf = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],\n                   'Col2': [2, 5, 8, 11, 14, 17],\n                   'Col3': [3, 6, 9, 12, 15, 18],\n                   'Type': [1, 1, 2, 2, 3, 3]})\nList = np.random.permutation(len(df))\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n"
    completion = GEN_SOLUTION_deepseek(base_checkpoint="deepseek-ai/deepseek-coder-1.3b-instruct", dp_epsilon='1e10', epoch=19, prompt=prompt)
