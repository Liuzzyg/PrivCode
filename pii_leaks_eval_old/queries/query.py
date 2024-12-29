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


def GEN_SOLUTION(base_checkpoint, model_checkpoint, dp_epsilon=None, step=None, prompt='', max_tokens=100, n=5, temperature=0.8):
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
        completion = tokenizer.decode(outputs[i], clean_up_tokenization_spaces=False)
        results.append(completion)
    # pdb.set_trace()
    print("generation is: \n", results)

    # pdb.set_trace()

    return results


if __name__ == '__main__':
    prompt = "{\n    \"facebook\": {\n        \"email\": \"abc@gmail.com\",\n        \"password\":"
    prompt = "def check_IP_address():\nIPv6_address = "
    # prompt = "Problem:\nI have the following DataFrame:\n    Col1  Col2  Col3  Type\n0      1     2     3     1\n1      4     5     6     1\n2      7     8     9     2\n3    10    11    12     2\n4    13    14    15     3\n5    16    17    18     3\n\n\nThe DataFrame is read from a CSV file. All rows which have Type 1 are on top, followed by the rows with Type 2, followed by the rows with Type 3, etc.\nI would like to shuffle the order of the DataFrame's rows according to a list. \nFor example, give a list [2, 4, 0, 3, 1, 5] and desired DataFrame should be:\n    Col1  Col2  Col3  Type\n2      7     8     9     2\n4     13    14    15     3\n0     1     2     3     1\n3    10    11    12     2\n1     4     5     6     1\n5    16    17    18     3\n...\nI want to know how many rows have different Type than the original DataFrame. In this case, 4 rows (0,1,2,4) have different Type than origin.\nHow can I achieve this?\n\n\nA:\n<code>\nimport pandas as pd\nimport numpy as np\n\n\ndf = pd.DataFrame({'Col1': [1, 4, 7, 10, 13, 16],\n                   'Col2': [2, 5, 8, 11, 14, 17],\n                   'Col3': [3, 6, 9, 12, 15, 18],\n                   'Type': [1, 1, 2, 2, 3, 3]})\nList = np.random.permutation(len(df))\n</code>\nresult = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n"
    results = GEN_SOLUTION(
        base_checkpoint="deepseek-ai/deepseek-coder-1.3b-instruct", 
        model_checkpoint='/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/pii_data/deepseek-coder-1.3b-instruct/dp10/checkpoint-500',
        # model_checkpoint='/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/pii_data/deepseek-coder-1.3b-instruct/dpinf/checkpoint-150',
        # dp_epsilon='1e10', 
        # step=19, 
        prompt=prompt
    )