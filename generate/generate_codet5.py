from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import pdb
import torch


def GEN_SOLUTION_Codet5p(base_checkpoint, dp_epsilon, prompt):
    if '2b' in base_checkpoint:
        # checkpoint = "Salesforce/codet5p-2b"
        device = "cuda" # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_checkpoint,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device)
        # pdb.set_trace()
        model = PeftModel.from_pretrained(
            base_model, 
            f"examples/starcoder/finetune/checkpoints/codet5p-2b/dp{dp_epsilon}/final_checkpoint",
            cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        )
        model = model.merge_and_unload().to(device)

        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        encoding['decoder_input_ids'] = encoding['input_ids'].clone()
        outputs = model.generate(**encoding, max_length=600)

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(completion)
        # ==> print "Hello World"
    else:
        device = "cuda" # for GPU usage or "cpu" for CPU usage

        tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
        base_model = T5ForConditionalGeneration.from_pretrained(base_checkpoint).to(device)
        model = PeftModel.from_pretrained(
            base_model, 
            f"examples/starcoder/finetune/checkpoints/codet5p-770m/dp{dp_epsilon}/final_checkpoint",
            cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        )
        model = model.merge_and_unload().to(device)

        # inputs = tokenizer.encode('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t"""\n<extra_id_0>', return_tensors="pt").to(device)
        # inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
        inputs = tokenizer.encode(prompt + '<extra_id_0>', return_tensors="pt").to(device)
        outputs = model.generate(
            inputs, 
            max_length=100
        )

        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(completion)
        # ==> print "Hello World"
    




    # # checkpoint = "Salesforce/codet5p-220m"
    # device = "cuda" # for GPU usage or "cpu" for CPU usage

    # tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    # # model = T5ForConditionalGeneration.from_pretrained(
    # #     'examples/starcoder/finetune/checkpoints/codet5p-220m/dp1e10_nolora_nodeepspeed/final_checkpoint'
    # #     ).to(device)
    # # model = T5ForConditionalGeneration.from_pretrained(
    # #     f'examples/starcoder/finetune/checkpoints/codet5p-770m/dp{dp_epsilon}_nolora/final_checkpoint'
    # #     ).to(device)
    # model = T5ForConditionalGeneration.from_pretrained(base_checkpoint).to(device)

    # # inputs = tokenizer.encode('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n\t""" Check if in given list of numbers, are any two numbers closer to each other than\n\tgiven threshold.\n\t>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n\tFalse\n\t>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n\tTrue\n\t"""\n<extra_id_0>', return_tensors="pt").to(device)
    # # inputs = tokenizer.encode("def print_hello_world():<extra_id_0>", return_tensors="pt").to(device)
    # inputs = tokenizer.encode(prompt+"<extra_id_0>", return_tensors="pt").to(device)
    # outputs = model.generate(inputs, max_length=100)

    # completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(completion)
    # # ==> print "Hello World"


    
    return completion

if __name__ == '__main__':
    GEN_SOLUTION_Codet5p("Salesforce/codet5p-2b", '1e-10', "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n")

