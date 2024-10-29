from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse
import json
import os

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

    # Save the merged model and tokenizer
    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
        tokenizer.push_to_hub(f"{args.base_model_name_or_path}-merged", use_temp_dir=False, private=True)
    else:
        model.save_pretrained(args.save_merged_model_path)
        tokenizer.save_pretrained(args.save_merged_model_path)
        print(f"Model saved to {args.save_merged_model_path}")

    # Modify tokenizer_config.json to add "chat_template"
    tokenizer_config_path = os.path.join(args.save_merged_model_path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r+") as file:
            tokenizer_config = json.load(file)
            # Add the "chat_template" field
            tokenizer_config["chat_template"] = (
                "{{bos_token}}{{'You are an exceptionally intelligent coding assistant that consistently delivers "
                "accurate and reliable responses to user instructions.\n\n'}}\n"
                "{%- for message in messages %}\n"
                "    {%- if message['role'] == 'system' %}\n"
                "        {{ raise_exception('System messages are not allowed in this template.') }}\n"
                "    {%- else %}\n"
                "        {%- if message['role'] == 'user' %}\n"
                "{{'Question: ' + message['content'] + '\n\n'}}\n"
                "        {%- else %}\n"
                "{{'Answer: ' + message['content'] + eos_token + '\n\n'}}\n"
                "        {%- endif %}\n"
                "    {%- endif %}\n"
                "{%- endfor %}\n"
                "{{'Answer: '}}"
            )
            # Save the updated configuration
            file.seek(0)
            json.dump(tokenizer_config, file, indent=4)
            file.truncate()
        print("Updated tokenizer_config.json with 'chat_template'.")

if __name__ == "__main__" :
    main()
