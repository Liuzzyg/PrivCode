import torch
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pdb
import os
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


class PostDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids']),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask']),
            'labels': torch.tensor(self.data[idx]['labels']),
        }


def prepare_sample_text_traindata(examples, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    texts = []
    for prompt, completion in zip(examples[input_column_name], examples[output_column_name]):
        # pdb.set_trace()
        text = completion
        texts.append(text)
    return texts

def prepare_sample_text_testdata(examples, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    texts = []
    for prompt, completion in zip(examples[input_column_name], examples[output_column_name]):
        # pdb.set_trace()
        text = prompt + completion
        texts.append(text)
    return texts

def get_metrics(label, score, fixed_fpr=0.01):
    """
    Compute TPR at FPR
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(label, score)
    tpr_at_low_fpr = tpr[np.where(fpr <= fixed_fpr)[0][-1]]
    return tpr_at_low_fpr


def forward_mia(model, data_loader, device, desc="Processing"):
    """
    Perform forward pass on data_loader and compute loss for each batch.
    """
    model = model.to(device)
    losses = []
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id

    for batch in tqdm(data_loader, desc=desc):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        # pdb.set_trace()
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for loss computation
        shift_logits = logits[..., :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous()      # (batch_size, seq_len-1)

        seq_lens = (shift_labels != pad_token_id).sum(dim=1)

        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction='none')  # (batch_size, seq_len-1)
        loss = loss.sum(dim=1) / seq_lens  # Normalize loss by sequence length
        losses.append(loss.mean().item()) 

    return torch.tensor(losses, device=device)


def get_dataloader(config, tokenizer):
    # Load training data
    dataset = load_dataset("json", data_files=config.train_data_path, split=config.split)
    
    # Split into training and testing sets
    dataset = dataset.train_test_split(train_size=0.95, seed=config.seed)
    train_data = dataset['train']

    # Load test data from a JSONL file
    test_data = load_dataset("json", data_files=config.test_data_path, split=config.split)

    test_data = test_data.train_test_split(train_size=0.5, seed=config.seed)
    train_data = test_data['train']
    test_data = test_data['test']

    # Limit sample size
    train_data = train_data.select(range(len(test_data)))

    print(f"Train data size: {len(train_data)}, Test data size: {len(test_data)}")

    def preprocess_function_traindata(examples):
        # pdb.set_trace()
        # buffer = prepare_sample_text_testdata(examples, 'problem', 'solution')
        buffer = prepare_sample_text_testdata(examples, 'prompt', 'generated_text')

        tokenized_inputs = tokenizer(buffer, truncation=False)
        
        result = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

        pad_token_id = tokenizer.eos_token_id

        for input_ids in tokenized_inputs["input_ids"]:
            input_ids += [tokenizer.eos_token_id]
            
            attention_mask = [1] * len(input_ids)
            
            for i in range(0, len(input_ids), config.seq_length):
                seq = input_ids[i : i + config.seq_length]
                mask = attention_mask[i : i + config.seq_length]

                if len(seq) < config.seq_length:
                    mask += [0] * (config.seq_length - len(seq))
                    seq += [pad_token_id] * (config.seq_length - len(seq))
                
                result["input_ids"].append(seq)
                result["labels"].append(seq)
                result["attention_mask"].append(mask)

        return result
    
    def preprocess_function_testdata(examples):
        # pdb.set_trace()
        buffer = prepare_sample_text_testdata(examples, 'prompt', 'generated_text')
        
        tokenized_inputs = tokenizer(buffer, truncation=False)
        
        result = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

        pad_token_id = tokenizer.eos_token_id

        for input_ids in tokenized_inputs["input_ids"]:
            input_ids += [tokenizer.eos_token_id]
            
            attention_mask = [1] * len(input_ids)
            
            for i in range(0, len(input_ids), config.seq_length):
                seq = input_ids[i : i + config.seq_length]
                mask = attention_mask[i : i + config.seq_length]

                if len(seq) < config.seq_length:
                    mask += [0] * (config.seq_length - len(seq))
                    seq += [pad_token_id] * (config.seq_length - len(seq))
                
                result["input_ids"].append(seq)
                result["labels"].append(seq)
                result["attention_mask"].append(mask)

        return result

    train_data = train_data.map(
        preprocess_function_traindata,
        batched=True,
        remove_columns=train_data.column_names,
        num_proc=None
    )
    # pdb.set_trace()

    test_data = test_data.map(
        preprocess_function_testdata,
        batched=True,
        remove_columns=test_data.column_names,
        num_proc=None
    )
    # pdb.set_trace()

    train_dataset = PostDataset(train_data)
    test_dataset = PostDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, test_loader


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_checkpoint)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_checkpoint,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
    )

    train_loader, test_loader = get_dataloader(config, tokenizer)

    model.eval()

    with torch.no_grad():
        test_losses = []
        train_losses = []

        # Repeat for stability
        for _ in range(config.repeat):
            # Compute loss for test and train data
            train_loss = forward_mia(model, train_loader, device, desc="Computing Train Loss")
            test_loss = forward_mia(model, test_loader, device, desc="Computing Test Loss")

            test_losses.append(test_loss)
            train_losses.append(train_loss)

        # Convert lists to tensors
        test_losses_tensor = torch.stack(test_losses)
        train_losses_tensor = torch.stack(train_losses)

        # Define a threshold for filtering (for example, 1e6)
        threshold = 1e3

        # Filter out infinity and values above the threshold
        valid_test_losses = test_losses_tensor[~torch.isinf(test_losses_tensor) & (test_losses_tensor < threshold)]
        valid_train_losses = train_losses_tensor[~torch.isinf(train_losses_tensor) & (train_losses_tensor < threshold)]

        # Check if there are any valid losses left for averaging
        if valid_test_losses.numel() == 0 or valid_train_losses.numel() == 0:
            print("No valid losses to compute average.")
            avg_test_loss = torch.tensor(float('nan'))  # or some appropriate value
            avg_train_loss = torch.tensor(float('nan'))  # or some appropriate value
        else:
            # Average the valid losses over the repeats
            avg_test_loss = torch.mean(valid_test_losses, dim=0)
            avg_train_loss = torch.mean(valid_train_losses, dim=0)

        # Create labels for TPR computation
        test_label = torch.ones(1, device=avg_test_loss.device)
        train_label = torch.zeros(1, device=avg_train_loss.device) 

        results = torch.stack([avg_test_loss, avg_train_loss]).cpu().detach().numpy()
        labels = torch.cat([test_label, train_label]).cpu().detach().numpy().astype(int)

        # Calculate TPR for different FPR thresholds
        tpr_at_low_fpr_1 = get_metrics(labels, results, fixed_fpr=0.1)
        tpr_at_low_fpr_2 = get_metrics(labels, results, fixed_fpr=0.01)
        tpr_at_low_fpr_3 = get_metrics(labels, results, fixed_fpr=0.001)
        tpr_at_low_fpr_4 = get_metrics(labels, results, fixed_fpr=0.2)

        # Print results
        result_str = (f"TPR@10%FPR: {tpr_at_low_fpr_1:.3f}, "
                      f"TPR@1%FPR: {tpr_at_low_fpr_2:.4f}, "
                      f"TPR@0.1%FPR: {tpr_at_low_fpr_3:.5f}, "
                      f"TPR@20%FPR: {tpr_at_low_fpr_4:.6f}")
        
        print(result_str)

        # Save results to a JSONL file
        results_to_save = {
            "TPR@10%FPR": tpr_at_low_fpr_1,
            "TPR@1%FPR": tpr_at_low_fpr_2,
            "TPR@0.1%FPR": tpr_at_low_fpr_3,
            "TPR@20%FPR": tpr_at_low_fpr_4
        }

        # Write to JSONL file
        with open(args.output_file, "a") as f:
            f.write(json.dumps(results_to_save) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")  # Add model checkpoint
    parser.add_argument("--model_checkpoint", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/original_data_merged/checkpoint-360")  # Add model checkpoint
    parser.add_argument("--train_data_path", type=str, default="data/private_syn/deepseek-coder-1.3b-instruct_original_data.jsonl")

    parser.add_argument("--test_data_path", type=str, default="pii_leaks_eval/queries/deepseek-coder-6.7b-base_dp10_syndata_step360_vllm.jsonl")
    parser.add_argument("--output_file", type=str, default="pii_leaks_eval/mia_results/deepseek-coder-6.7b-base_dp10_syndata_step360.jsonl")

    # parser.add_argument("--test_data_path", type=str, default="pii_leaks_eval/queries/deepseek-coder-6.7b-base_original_data_step360_vllm.jsonl")
    # parser.add_argument("--output_file", type=str, default="pii_leaks_eval/mia_results/deepseek-coder-6.7b-base_original_data_step360.jsonl")

    # # 130 step
    # parser.add_argument("--model_checkpoint", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/original_data_merged/checkpoint-130")  # Add model checkpoint
    # parser.add_argument("--train_data_path", type=str, default="data/private_syn/deepseek-coder-1.3b-instruct_original_data.jsonl")

    # # parser.add_argument("--test_data_path", type=str, default="pii_leaks_eval/queries/deepseek-coder-6.7b-base_dp10_syndata_step130_vllm.jsonl")
    # # parser.add_argument("--output_file", type=str, default="pii_leaks_eval/mia_results/deepseek-coder-6.7b-base_dp10_syndata_step130.jsonl")

    # parser.add_argument("--test_data_path", type=str, default="pii_leaks_eval/queries/deepseek-coder-6.7b-base_original_data_step130_vllm.jsonl")
    # parser.add_argument("--output_file", type=str, default="pii_leaks_eval/mia_results/deepseek-coder-6.7b-base_original_data_step130.jsonl")




    parser.add_argument("--seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sigma", type=list, default=[0.05, 0.01])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(config=args)
