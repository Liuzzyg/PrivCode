import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import PeftModel
import numpy as np
import argparse
import pdb
import copy

parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct", help="which results to run")
# parser.add_argument("--model", type=str, default="Salesforce/codet5p-770m", help="which results to run")
parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-6.7b-instruct", help="which results to run")
parser.add_argument('--dp_epsilon', type=str, default='1e-10', help="the epsilon value for differential privacy")

args = parser.parse_args()
model_name = args.model

def compare_model_weights(base_model, merged_model):
    # Get the state dictionaries of both models
    base_state_dict = base_model.state_dict()
    merged_state_dict = merged_model.state_dict()

    # Keep track of layers that have changed
    changed_layers = []

    for name, base_param in base_state_dict.items():
        merged_param = merged_state_dict[name]
        
        # Check if the parameters are different (i.e., after applying LoRA)
        if not torch.equal(base_param, merged_param):
            # Calculate the absolute difference between parameters
            diff = (base_param - merged_param).abs()
            max_diff = diff.max().item()  # Get the maximum difference
            mean_diff = diff.mean().item()  # Get the mean difference

            changed_layers.append({
                "layer": name,
                "max_diff": max_diff,
                "mean_diff": mean_diff
            })
    
    # Print out the changed layers and the differences
    if changed_layers:
        print(f"{len(changed_layers)} layers have changed:")
        for layer_info in changed_layers:
            print(f"Layer: {layer_info['layer']}, Max diff: {layer_info['max_diff']}, Mean diff: {layer_info['mean_diff']}")
    else:
        print("No layers have changed.")

# Load base model (before LoRA merging)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, resume_download=True,
    cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
    use_auth_token=True,
)

copy_model = copy.deepcopy(base_model)

# Load LoRA fine-tuned model and merge it
model = PeftModel.from_pretrained(
    base_model, 
    # f"examples/starcoder/finetune/checkpoints/deepseek-coder-1.3b-instruct/dp{dp_epsilon}/final_checkpoint"
    # f"examples/starcoder/finetune/checkpoints/deepseek-coder-6.7b-instruct/dp{dp_epsilon}/final_checkpoint"
    '/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/magicoder/deepseek-coder-6.7b-instruct/dp10/checkpoint-110'
    # f'examples/starcoder/finetune/checkpoints/synthetic_numpy/deepseek-coder-1.3b-instruct/dp{args.dp_epsilon}/final_checkpoint'
)

# Merge LoRA weights into the base model
model = model.merge_and_unload()

# model_1e10 = AutoModelForSeq2SeqLM.from_pretrained(
#     'examples/starcoder/finetune/checkpoints/alpaca/codet5p-220m/dp1e10_nolora_nodeepspeed/final_checkpoint',
#     resume_download=True,
#     # cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
#     use_auth_token=True,
# )
# model_1e_10 = AutoModelForSeq2SeqLM.from_pretrained(
#     'examples/starcoder/finetune/checkpoints/alpaca/codet5p-220m/dp1e-10_nolora_nodeepspeed/final_checkpoint',
#     resume_download=True,
#     # cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
#     use_auth_token=True,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     '/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/alpaca_step/deepseek-coder-1.3b-instruct/dp10_nolora_1norm/checkpoint-70',
#     # resume_download=True,
#     # cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
#     # use_auth_token=True,
# )


compare_model_weights(copy_model, model)

# pdb.set_trace()
# Compare weights between base model and merged model
compare_model_weights(base_model, model)
