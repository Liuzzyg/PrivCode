import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset
from scipy.stats import entropy
import numpy as np
from tqdm import tqdm  # 进度条库
import pdb

# 加载模型和tokenizer
def load_model_and_tokenizer(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
        use_auth_token=True,
    ).to(device)  # 将模型转移到 GPU

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer, model


# 批量获取 logits 并打印维度信息
def get_logits_batch(model, tokenizer, batch_input_ids, batch_attention_mask, device):
    # 将输入数据转移到 GPU 上
    inputs = {
        "input_ids": torch.tensor(batch_input_ids).to(device),
        "attention_mask": torch.tensor(batch_attention_mask).to(device)
    }

    with torch.no_grad():
        output = model(**inputs)
    logits = output.logits  # (batch_size, sequence_length, vocab_size)

    return logits.cpu()  # 将 logits 转回 CPU 以便后续计算


# 批量分析函数，计算熵和其他分布差异
def analyze_batch_logits(logits):
    # 计算每个token的熵
    batch_size, seq_len, vocab_size = logits.shape

    # 对序列维度进行平均，获取每个 token 的平均 logits
    logits_avg = logits.mean(dim=1)  # (batch_size, vocab_size)
    
    # 计算批量熵
    entropies = []
    for i in range(batch_size):
        token_distribution = torch.softmax(logits_avg[i], dim=-1)
        entropies.append(entropy(token_distribution.numpy(), axis=-1))

    return np.array(entropies)


# 批量分析整个数据集的函数
def analyze_dataset(model, tokenizer, dataset, batch_size=16, device='cpu'):
    all_entropies = []
    # 添加进度条
    for i in tqdm(range(0, len(dataset), batch_size), desc="Analyzing dataset"):
        # 获取当前批次的数据
        batch = dataset[i: i + batch_size]

        # 直接使用预处理后的 input_ids 和 attention_mask
        batch_input_ids = batch["input_ids"]
        batch_attention_mask = batch["attention_mask"]
        
        # 获取 logits
        logits = get_logits_batch(model, tokenizer, batch_input_ids, batch_attention_mask, device)
        
        # 计算熵
        batch_entropies = analyze_batch_logits(logits)
        
        # 收集结果
        all_entropies.extend(batch_entropies)

    return np.array(all_entropies)


def prepare_sample_text(examples, column_name):
    """Prepare the text from a sample of the dataset."""
    texts = []
    for text in examples[column_name]:
        texts.append(text)
    return texts


# 加载数据集并进行预处理
def load_and_preprocess_dataset(tokenizer, seq_length, dataset_name='terryyz/pii', split='', column_name='', split_ratio=0.99999):
    dataset = load_dataset(
        dataset_name,
        split=split,
        use_auth_token=True,
        cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
    )

    # Split the dataset into train and validation sets
    dataset = dataset.train_test_split(train_size=split_ratio, seed=42)
    train_data = dataset['train']
    valid_data = dataset['test']

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    def preprocess_function(examples):
        buffer = prepare_sample_text(examples, column_name)
        tokenized_inputs = tokenizer(buffer, truncation=True)
        
        result = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

        pad_token_id = tokenizer.eos_token_id

        for input_ids in tokenized_inputs["input_ids"]:
            input_ids += [tokenizer.eos_token_id]
            
            attention_mask = [1] * len(input_ids)
            
            for i in range(0, len(input_ids), seq_length):
                seq = input_ids[i : i + seq_length]
                mask = attention_mask[i : i + seq_length]

                if len(seq) < seq_length:
                    mask += [0] * (seq_length - len(seq))
                    seq += [pad_token_id] * (seq_length - len(seq))
                
                result["input_ids"].append(seq)
                result["labels"].append(seq)
                result["attention_mask"].append(mask)

        return result

    # Apply preprocessing to the train dataset
    train_dataset = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        num_proc=4
    )
    return train_dataset


# 比较两个数据集的熵分布差异
def compare_two_datasets(model_code, tokenizer_code, dataset_code, model_text, tokenizer_text, dataset_text, device):
    # 分别分析两个数据集的熵
    entropies_text = analyze_dataset(model_text, tokenizer_text, dataset_text, device=device)
    entropies_code = analyze_dataset(model_code, tokenizer_code, dataset_code, device=device)

    # 打印统计结果
    print(f"Dataset_code Entropy Mean: {np.mean(entropies_code)}, Std: {np.std(entropies_code)}")
    print(f"Dataset_text Entropy Mean: {np.mean(entropies_text)}, Std: {np.std(entropies_text)}")

    # 计算熵的总体差异
    entropy_diff = np.mean(entropies_code) - np.mean(entropies_text)
    print(f"Entropy Difference (Dataset_code - Dataset_text): {entropy_diff}")


# 主函数执行整个流程
def main(model_path_code, model_path_text, seq_length=512, batch_size=16):
    # 检查是否有 GPU 可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载两个模型和 tokenizer
    tokenizer_code, model_code = load_model_and_tokenizer(model_path_code, device)
    tokenizer_text, model_text = load_model_and_tokenizer(model_path_text, device)

    # 加载并预处理两个数据集（一个代码数据集，一个文本数据集）
    dataset_code = load_and_preprocess_dataset(tokenizer_code, seq_length, dataset_name='ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', column_name='solution', split_ratio=0.06648936)  # 5k
    # dataset_code = load_and_preprocess_dataset(tokenizer_code, seq_length, dataset_name='terryyz/pii', column_name='text', split='test', split_ratio=0.413223)  # 5k
    dataset_text = load_and_preprocess_dataset(tokenizer_text, seq_length, dataset_name='ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', column_name='problem', split_ratio=0.06648936)  # 5k
    # dataset_text = load_and_preprocess_dataset(tokenizer_text, seq_length, dataset_name='pig4431/yelp_train25k_test5k_valid5k', split='test', column_name='text', split_ratio=0.99999)  # 5k
    # dataset_code = load_and_preprocess_dataset(tokenizer_code, seq_length, dataset_name='ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', column_name='solution', split_ratio=0.0001)  # 5k
    # # dataset_code = load_and_preprocess_dataset(tokenizer_code, seq_length, dataset_name='terryyz/pii', split='test', column_name='text', split_ratio=0.1)  # 5k
    # dataset_text = load_and_preprocess_dataset(tokenizer_text, seq_length, dataset_name='pig4431/yelp_train25k_test5k_valid5k', split='test', column_name='text', split_ratio=0.01)  # 5k

    # 比较两个数据集的熵分布
    compare_two_datasets(model_code, tokenizer_code, dataset_code, model_text, tokenizer_text, dataset_text, device)

# 调用主函数
main('deepseek-ai/deepseek-coder-1.3b-base', 'openai-community/gpt2')
