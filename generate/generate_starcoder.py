import torch
import pdb

from safetensors.torch import load_file
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm



def GEN_SOLUTION_starcoder(base_checkpoint, dp_epsilon, prompt):
    # single sample test

    # checkpoint = "bigcode/starcoder2-3b"
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        use_auth_token=True,
    )

    model = PeftModel.from_pretrained(
        base_model, 
        # f"examples/starcoder/finetune/checkpoints/starcoder2-3b_c_attn/dp{dp_epsilon}/final_checkpoint"
        f'examples/starcoder/finetune/checkpoints/alpaca_new/starcoder2-3b/dp{dp_epsilon}/final_checkpoint'
    )

    model = model.merge_and_unload().to(device)

    # model = base_model.to(device)

    # inputs = tokenizer.encode('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t"""\n', return_tensors="pt").to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # pdb.set_trace()

    outputs = model.generate(
        inputs, 
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        # top_p=0.95,
        # temperature=0.8,
        # do_sample=True,
    )
    # clean_up_tokenization_spaces=False prevents a tokenizer edge case which can result in spaces being removed around punctuation
    # pdb.set_trace()
    completion = tokenizer.decode(outputs[0][len(inputs[0]):], clean_up_tokenization_spaces=False)
    print(completion)

    return completion


if __name__ == '__main__':
    prompt = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
    completion = GEN_SOLUTION_starcoder(base_checkpoint="bigcode/starcoder2-3b", dp_epsilon='1e-10', prompt=prompt)
