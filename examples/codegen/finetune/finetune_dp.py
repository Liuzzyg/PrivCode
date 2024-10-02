import argparse
import os
import json
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training , set_peft_model_state_dict
import torch.distributed
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
# from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from fastDP import PrivacyEngine

from trainer_fp16 import Trainer
# from trainer import Trainer
from compiled_args import (PrivacyArguments, TrainingArguments)
"""
Fine-Tune StarCoder on Code Alpaca/SE
"""


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control
    

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="Qwen/CodeQwen1.5-7B")
    parser.add_argument("--model_path", type=str, default="bigcode/starcoder2-3b")
    # parser.add_argument("--dataset_name", type=str, default="bigcode/starcoderdata")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/CodeAlpaca_20K")
    # parser.add_argument("--subset", type=str, default="python")
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=1000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)

    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str, default="completion")

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--multi_gpus", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="examples/starcoder/finetune/checkpoints/CodeAlpaca_20K-starcoder2-3b_dp10_fp16")
    parser.add_argument("--log_freq", default=100, type=int)
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    parser.add_argument("--save_freq_epoch", default=1, type=int)

    # DP
    parser.add_argument("--per_example_max_grad_norm", type=float, default=0.1)
    parser.add_argument("--target_epsilon", type=float, default=10)
    parser.add_argument("--target_delta", type=float, default=1e-5)
    parser.add_argument("--non_private", type=str, default='no')
    parser.add_argument("--clipping_mode", type=str, default='ghost')

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, input_column_name="prompt", output_column_name="completion", nb_examples=400):
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


def prepare_sample_text(example, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example[input_column_name]}\n\nAnswer: {example[output_column_name]}"
    return text


# class ConstantLengthDataset(IterableDataset):
#     """
#     Iterable dataset that returns constant length chunks of tokens from stream of text files.
#         Args:
#             tokenizer (Tokenizer): The processor used for proccessing the data.
#             dataset (dataset.Dataset): Dataset with text files.
#             infinite (bool): If True the iterator is reset after dataset reaches end else stops.
#             seq_length (int): Length of token sequences to return.
#             num_of_sequences (int): Number of token sequences to keep in buffer.
#             chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
#     """

#     def __init__(
#         self,
#         tokenizer,
#         dataset,
#         infinite=False,
#         seq_length=1024,
#         num_of_sequences=1024,
#         chars_per_token=3.6,
#         input_column_name="prompt",
#         output_column_name="completion"
#     ):
#         self.tokenizer = tokenizer
#         self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
#         self.dataset = dataset
#         self.seq_length = seq_length
#         self.infinite = infinite
#         self.current_size = 0
#         self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
#         self.input_column_name = input_column_name
#         self.output_column_name = output_column_name

#     def __iter__(self):
#         iterator = iter(self.dataset)
#         more_examples = True
#         while more_examples:
#             buffer, buffer_len = [], 0
#             while True:
#                 if buffer_len >= self.max_buffer_size:
#                     break
#                 try:
#                     buffer.append(prepare_sample_text(next(iterator), self.input_column_name, self.output_column_name))
#                     buffer_len += len(buffer[-1])
#                 except StopIteration:
#                     if self.infinite:
#                         iterator = iter(self.dataset)
#                     else:
#                         more_examples = False
#                         break
#             tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
#             all_token_ids = []
#             for tokenized_input in tokenized_inputs:
#                 all_token_ids.extend(tokenized_input + [self.concat_token_id])
#             for i in range(0, len(all_token_ids), self.seq_length):
#                 input_ids = all_token_ids[i : i + self.seq_length]
#                 if len(input_ids) == self.seq_length:
#                     self.current_size += 1
#                     yield {
#                         "input_ids": torch.LongTensor(input_ids),
#                         "labels": torch.LongTensor(input_ids),
#                     }

def create_datasets(tokenizer, args):
    if args.streaming:
        dataset = load_dataset(
            args.dataset_name,
            data_dir=args.subset,
            split=args.split,
            use_auth_token=True,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )

        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        dataset = load_dataset(
            args.dataset_name,
            data_dir=args.subset,
            # split=args.split,
            use_auth_token=True,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.input_column_name, args.output_column_name)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_data,
    #     infinite=True,
    #     seq_length=args.seq_length,
    #     chars_per_token=chars_per_token,
    #     input_column_name=args.input_column_name,
    #     output_column_name=args.output_column_name
    # )
    # valid_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     valid_data,
    #     infinite=False,
    #     seq_length=args.seq_length,
    #     chars_per_token=chars_per_token,
    #     input_column_name=args.input_column_name,
    #     output_column_name=args.output_column_name
    # )

    def preprocess_function(examples):
        buffer = prepare_sample_text(examples, args.input_column_name, args.output_column_name)
        tokenized_inputs = tokenizer(buffer, truncation=False)["input_ids"]
        all_token_ids = []
        for tokenized_input in tokenized_inputs:
            all_token_ids.extend([tokenized_input] + [tokenizer.eos_token_id])
        
        result = {
            "input_ids": [],
            "labels": []
        }
        
        for i in range(0, len(all_token_ids), args.seq_length):
            input_ids = all_token_ids[i : i + args.seq_length]
            if len(input_ids) == args.seq_length:
                result["input_ids"].append(input_ids)
                result["labels"].append(input_ids)
        
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

    # return train_dataset, valid_dataset, len(dataset) - args.size_valid_set
    return train_dataset, valid_dataset, len(train_data)


def run_training(args, tokenizer, train_data, val_data, total_train_data_length):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_8bit=True,
        # device_map="auto",
        device_map={"": Accelerator().process_index},
        # device_map={'':torch.cuda.current_device()}
    )
    print(type(model))
    
    model = prepare_model_for_kbit_training(model)
    print(type(model))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["c_proj", "c_attn", "q_attn"]
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
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name="StarCoder-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        disable_tqdm=False
    )

    privacy_args = PrivacyArguments(
        per_example_max_grad_norm=args.per_example_max_grad_norm,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        non_private=args.non_private,
        clipping_mode=args.clipping_mode 
    )

    if not args.multi_gpus:
        torch.distributed.init_process_group(backend='nccl')

    # Hacky way to set noise_multiplier.
    privacy_engine = PrivacyEngine(
        module=model,
        batch_size=training_args.train_batch_size,
        sample_size=total_train_data_length,
        epochs=training_args.num_train_epochs,
        max_grad_norm=privacy_args.per_example_max_grad_norm,
        noise_multiplier=privacy_args.noise_multiplier,
        target_epsilon=privacy_args.target_epsilon,
        target_delta=privacy_args.target_delta,
        accounting_mode=privacy_args.accounting_mode,
        clipping_mode=privacy_args.clipping_mode,
        clipping_fn=privacy_args.clipping_fn,
        clipping_style=privacy_args.clipping_style,
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
        # tokenizer=tokenizer, 
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

    # # Setup learning rate scheduler
    # if training_args.logical_batch_size!=None:
    #     trainer.args.gradient_accumulation_steps=training_args.logical_batch_size/training_args.per_device_train_batch_size/num_GPUs
    # else:
    #     training_args.logical_batch_size=trainer.args.gradient_accumulation_steps*training_args.per_device_train_batch_size*num_GPUs

    # num_update_steps_per_epoch = len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    # num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    # t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)

    # if training_args.lr_decay:
    #     trainer.lr_scheduler = get_linear_schedule_with_warmup(
    #         trainer.optimizer,
    #         num_warmup_steps=training_args.warmup_steps,
    #         num_training_steps=t_total,
    #     )
    # else:
    #     trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)


    print('privacy_args: ')
    print(json.dumps(privacy_args.__dict__, indent=4))
    privacy_engine.attach(optimizer)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",)
    train_dataset, eval_dataset, total_train_data_length = create_datasets(tokenizer, args)
    run_training(args, tokenizer, train_dataset, eval_dataset, total_train_data_length)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    if not args.multi_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'

    main(args)
