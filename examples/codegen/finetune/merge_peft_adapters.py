from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import pdb
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument("--peft_model_path", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dp10_syndata/checkpoint-130")
    parser.add_argument("--save_merged_model_path", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dp10_syndata_merged/checkpoint-130")
    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()

def main():
    args = get_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        # return_dict=True,
        # torch_dtype=torch.float16 
    )

    model = PeftModel.from_pretrained(base_model, args.peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
        tokenizer.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    else:
        model.save_pretrained(args.save_merged_model_path)
        tokenizer.save_pretrained(args.save_merged_model_path)
        print(f"Model saved to {args.save_merged_model_path}")

if __name__ == "__main__" :
    main()
