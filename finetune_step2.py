import argparse
import os
import json
import torch
import pdb
import pandas as pd
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, Features, Value
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training , set_peft_model_state_dict
import torch.distributed
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
# from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from fastDP import PrivacyEngine, PrivacyEngine_Distributed_Stage_2_and_3

from trainer_deepspeed_dpbaseline import Trainer
from compiled_args import (PrivacyArguments, TrainingArguments)
from deepspeed.ops.adam import DeepSpeedCPUAdam


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")
    # parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    # parser.add_argument("--model_path", type=str, default="Qwen/CodeQwen1.5-7B")
    # parser.add_argument("--model_path", type=str, default="bigcode/starcoder2-3b")
    # parser.add_argument("--dataset_name", type=str, default="liuzzyg/private-syndata")
    # parser.add_argument("--dataset_name", type=str, default="liuzzyg/magi-original-data")
    parser.add_argument("--dataset_name", type=str, default="data/private_syn/deepseek-coder-1.3b-instruct_original_data.jsonl")
    # parser.add_argument("--subset", type=str, default="python")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=1000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    # # # magicoder oss instruct data
    parser.add_argument("--input_column_name", type=str, default="problem")
    parser.add_argument("--output_column_name", type=str, default="solution")
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--deepspeed_config", type=str, default='examples/codegen/finetune/config_stage1_baseline.json')
    parser.add_argument("--multi_gpus", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    # parser.add_argument("--output_dir", type=str, default="examples/starcoder/finetune/checkpoints/starcoderdata_numpy/starcoder2-3b/dp1e-10")
    # parser.add_argument("--output_dir", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints/valid_synthetic_numpy/deepseek-coder-1.3b-instruct/nodp_nolora_1norm")
    # parser.add_argument("--output_dir", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dp10_step800_test")
    parser.add_argument("--output_dir", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dpsgd_baseline")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument("--save_freq_epoch", default=1, type=int)

    # DP
    parser.add_argument("--logical_batch_size", type=int, default=256)
    parser.add_argument("--per_example_max_grad_norm", type=float, default=1)
    parser.add_argument("--target_epsilon", type=float, default=10)
    parser.add_argument("--target_delta", type=float, default=1e-5)
    parser.add_argument("--non_private", type=str, default='no')
    # parser.add_argument("--non_private", type=str, default='y')
    parser.add_argument("--clipping_mode", type=str, default='ghost')

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, input_column_name="prompt", output_column_name="completion", nb_examples=400): # alpaca, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


# def chars_token_ratio(dataset, tokenizer, column_name='content', nb_examples=400):    #input_column_name="prompt", output_column_name="completion", nb_examples=400): # alpaca
#     """
#     Estimate the average number of characters per token in the dataset.
#     """
#     total_characters, total_tokens = 0, 0
#     for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
#         text = prepare_sample_text(example, column_name)
#         total_characters += len(text)
#         if tokenizer.is_fast:
#             total_tokens += len(tokenizer(text).tokens())
#         else:
#             total_tokens += len(tokenizer.tokenize(text))

#     return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def prepare_sample_text(examples, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    texts = []
    for prompt, completion in zip(examples[input_column_name], examples[output_column_name]):
        # pdb.set_trace()
        text = f"Question: {prompt}\n\nAnswer: {completion}"
        texts.append(text)
    return texts

def prepare_sample_text_pii(examples):
    """Prepare the text from a sample of the dataset."""
    texts = []
    for text in examples['text']:
        # pdb.set_trace()
        texts.append(text)
    return texts


# def prepare_sample_text(examples, input_column_name="prompt", output_column_name="completion"):
#     """Prepare the text from a sample of the dataset."""
#     texts = []
#     for prompt, completion in zip(examples[input_column_name], examples[output_column_name]):
#         # pdb.set_trace()
#         text = f"API: Numpy\n\nAnswer: {completion}"
#         texts.append(text)
#     return texts


def create_datasets(tokenizer, args):
    if args.dataset_name == 'ise-uiuc/Magicoder-OSS-Instruct-75K':
        dataset = load_dataset(
            args.dataset_name,
            data_dir=args.subset,
            split=args.split,
            use_auth_token=True,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
            cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
        )
        # only train split
        dataset = dataset.train_test_split(test_size=0.74, seed=args.seed)
        train_data = dataset['train']
        valid_data = dataset['test']

        args.input_column_name = "problem"
        args.output_column_name = "solution"

    elif args.dataset_name == 'data/oss_instruction/valid_processed_instruction_data_25k.jsonl':
        # synthetic private api numpy
        dataset = load_dataset(
            "json", 
            data_files=args.dataset_name,
            split=args.split
        )
        dataset = dataset.train_test_split(train_size=0.74, seed=args.seed)
        train_data = dataset['train']
        valid_data = dataset['test']
        
        args.input_column_name = "Problem"
        args.output_column_name = "Solution"
        
    # elif args.dataset_name == 'pii_leaks_eval/pii_dataset/pii_dataset.jsonl':
    #     dataset = load_dataset(
    #         "json", 
    #         data_files=args.dataset_name,
    #         split=args.split
    #     )
    elif args.dataset_name == 'terryyz/pii':
        dataset = load_dataset(
            args.dataset_name,
            split='test',
            use_auth_token=True,
            cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
        )
        # only train split
        dataset = dataset.train_test_split(train_size=0.99999, seed=args.seed)
        train_data = dataset['train']
        valid_data = dataset['test']

        args.input_column_name = "problem"
        args.output_column_name = "solution"

    # step2
    else:
        dataset = load_dataset(
            "json", 
            data_files=args.dataset_name,
            split=args.split
        )
        dataset = dataset.train_test_split(train_size=0.99999, seed=args.seed)
        train_data = dataset['train']
        valid_data = dataset['test']

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    # pdb.set_trace()

    def preprocess_function(examples):
        # if args.dataset_name == 'pii_leaks_eval/pii_dataset/pii_dataset.jsonl':
        if args.dataset_name == 'terryyz/pii':
            buffer = prepare_sample_text_pii(examples)
        else:
            buffer = prepare_sample_text(examples, args.input_column_name, args.output_column_name)

        tokenized_data = tokenizer(buffer, truncation=False, padding=False, add_special_tokens=False)

        result = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

        pad_token_id = tokenizer.eos_token_id
        answer_marker = tokenizer.encode("Answer: ", add_special_tokens=False)

        for input_ids in tokenized_data["input_ids"]:
            # Locate the split point between the prompt and completion
            prompt_end_idx = input_ids.index(answer_marker[0]) + 1


            # Generate labels
            labels = [-100] * prompt_end_idx + input_ids[prompt_end_idx:]
            
            # Add EOS token
            input_ids.append(pad_token_id)
            labels.append(pad_token_id)

            # Create attention mask
            attention_mask = [1] * len(input_ids)

            # Split into chunks of args.seq_length
            for i in range(0, len(input_ids), args.seq_length):
                seq = input_ids[i : i + args.seq_length]
                lbl = labels[i : i + args.seq_length]
                mask = attention_mask[i : i + args.seq_length]

                # Handle padding for sequences shorter than args.seq_length
                if len(seq) < args.seq_length:
                    padding_length = args.seq_length - len(seq)
                    seq += [pad_token_id] * padding_length
                    lbl += [-100] * padding_length
                    mask += [0] * padding_length

                result["input_ids"].append(seq)
                result["labels"].append(lbl)
                result["attention_mask"].append(mask)

        return result

    train_dataset = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        num_proc=args.num_workers
    )

    valid_dataset = valid_data.map(
        preprocess_function,
        batched=True,
        remove_columns=valid_data.column_names,
        num_proc=args.num_workers
    )

    # pdb.set_trace()
    return train_dataset, valid_dataset, len(train_data)
    pdb.set_trace()


def run_training(args, tokenizer, train_data, val_data, total_train_data_length):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        use_auth_token=True,
        # use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        # device_map="auto",
        device_map={"": Accelerator().process_index},
        # device_map={'':torch.cuda.current_device()}
    )
    print(type(model))
    
    model = prepare_model_for_kbit_training(model)
    print(type(model))
    # pdb.set_trace()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules = ["c_proj", "c_attn", "q_attn"]  # starcoder
        target_modules = [
            "self_attn.q_proj", 
            "self_attn.k_proj", 
            "self_attn.v_proj", 
            "self_attn.o_proj",
            "mlp.gate_proj", 
            "mlp.up_proj", 
            "mlp.down_proj"
        ]  # codeqwen  deepseek-coder  starcoder
        # target_modules = ["qkv_proj"]   # Phi-3.5-mini
    )

    model = get_peft_model(model, lora_config)
    print(type(model))

    print_trainable_parameters(model)

    train_data.start_iteration = 0
    # accelerator = Accelerator()
    num_GPUs = torch.cuda.device_count()
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Number of GPUs available: {num_GPUs}")

    # train_data = accelerator.prepare(train_data)
    # val_data = accelerator.prepare(val_data)
    # model = accelerator.prepare(model)

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        evaluate_before_training='no',
        save_strategy="steps",
        load_best_model_at_end=True,
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        num_train_epochs=args.epochs,
        save_steps=args.save_freq,
        save_epochs=args.save_freq_epoch,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="args.wandb_name",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
        deepspeed_config=args.deepspeed_config
    )

    privacy_args = PrivacyArguments(
        per_example_max_grad_norm=args.per_example_max_grad_norm,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        non_private=args.non_private,
        clipping_mode=args.clipping_mode 
    )

    if 'stage1' in args.deepspeed_config:
        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=args.logical_batch_size,
            sample_size=total_train_data_length,
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            non_private=privacy_args.non_private,
            accounting_mode=privacy_args.accounting_mode,
            clipping_mode=privacy_args.clipping_mode,
            clipping_fn=privacy_args.clipping_fn,
            clipping_style='layer-wise',
            # clipping_style=privacy_args.clipping_style,
            # origin_params=origin_params,
            num_GPUs=num_GPUs,
            torch_seed_is_fixed=True,
        )
    else:        
        privacy_engine = PrivacyEngine_Distributed_Stage_2_and_3(
            module=model,
            batch_size=args.logical_batch_size,
            sample_size=total_train_data_length,
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            non_private=privacy_args.non_private,
            accounting_mode=privacy_args.accounting_mode,
            clipping_mode=privacy_args.clipping_mode,
            clipping_fn=privacy_args.clipping_fn,
            clipping_style='layer-wise',
            # clipping_style=privacy_args.clipping_style,
            # origin_params=origin_params,
            num_GPUs=num_GPUs,
            torch_seed_is_fixed=True,
        )

    # Originally, these could have been null.
    privacy_args.noise_multiplier = privacy_engine.noise_multiplier
    privacy_args.target_delta = privacy_engine.target_delta

    # trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback])
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        privacy_args=privacy_args,
        train_dataset=train_data, 
        eval_dataset=val_data
        )


    # Initialize the optimizer
    params = model.parameters()

    optimizer = torch.optim.AdamW(
        params=params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    trainer.optimizer = optimizer

    print('privacy_args: ')
    print(json.dumps(privacy_args.__dict__, indent=4))
    if not training_args.deepspeed_config:
        privacy_engine.attach(optimizer)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    train_dataset, eval_dataset, total_train_data_length = create_datasets(tokenizer, args)
    run_training(args, tokenizer, train_dataset, eval_dataset, total_train_data_length)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
