import torch
import pdb

from safetensors.torch import load_file
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm



def GEN_SOLUTION_CodeQwen(base_checkpoint, dp_epsilon, prompt):
    # single sample test
    # pdb.set_trace()
    device = "cuda" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        use_auth_token=True,
    )

    model = PeftModel.from_pretrained(
        base_model, 
        f"examples/starcoder/finetune/checkpoints/CodeQwen1.5-7B/dp{dp_epsilon}/final_checkpoint"
    )

    model = model.merge_and_unload().to(device)
    model.eval()
    # model = base_model.to(device)

    # inputs = tokenizer.encode('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t"""\n', return_tensors="pt").to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs, 
        max_length=600,
        eos_token_id=tokenizer.eos_token_id,
        # top_p=0.95,
        # temperature=0.8,
        # do_sample=False,
    )
    # clean_up_tokenization_spaces=False prevents a tokenizer edge case which can result in spaces being removed around punctuation
    completion = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)
    print(completion)
    # pdb.set_trace()

    return completion

if __name__ == '__main__':
    prompt = '\ndef generate_integers(a, b):\n    """\n    Given two positive integers a and b, return the even digits between a\n    and b, in ascending order.\n\n    For example:\n    generate_integers(2, 8) => [2, 4, 6, 8]\n    generate_integers(8, 2) => [2, 4, 6, 8]\n    generate_integers(10, 14) => []\n    """\n'
    completion = GEN_SOLUTION_CodeQwen(base_checkpoint="Qwen/CodeQwen1.5-7B", dp_epsilon='1e-10', prompt=prompt)
