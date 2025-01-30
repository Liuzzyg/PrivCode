import argparse
import json
from collections import defaultdict
import code_bert_score
from tqdm import tqdm  # 引入tqdm库


def create_problem_language_mapping(original_data):
    """根据原始数据创建问题-语言映射表"""
    problem_language_map = {}
    for sample in original_data:
        problem = sample.get("problem", "").strip().lower()
        language = sample.get("lang", "unknown").lower()
        problem_language_map[problem] = language
    return problem_language_map


def group_generated_by_language(generated_data, problem_language_map):
    """根据问题语言映射表对生成的数据按语言分组"""
    grouped_data = defaultdict(list)
    for sample in tqdm(generated_data, desc="Grouping generated data by language", unit="sample"):  # 添加进度条
        problem = sample.get("problem", "").strip().lower()
        language = problem_language_map.get(problem, "unknown")
        if language != "unknown":  # 只保留匹配到语言的样本
            grouped_data[language].append(sample)
    return grouped_data


def filter_by_similarity(grouped_generated, grouped_original, threshold):
    """按相似度过滤生成样本"""
    filtered_samples = []

    # 需要进行相似度评估的语言
    languages_to_filter = {"python", "java", "cpp"}

    for language, generated_samples in tqdm(grouped_generated.items(), desc="Filtering by similarity", unit="language"):  # 添加进度条
        # 获取对应语言的原始样本
        original_samples = grouped_original.get(language, [])
        if not original_samples:
            print(f"No original samples for language: {language}")
            continue

        generated_codes = [sample["solution"] for sample in generated_samples]
        original_codes = [sample["solution"] for sample in original_samples]

        print(f"Processing language: {language} with {len(generated_samples)} samples")

        # 如果是需要评估相似度的语言，进行相似度计算
        if language in languages_to_filter:
            # 计算相似度
            pred_results = code_bert_score.score(cands=generated_codes, refs=original_codes, lang=language)
            f1_scores = pred_results[2].numpy()

            # 筛选出相似度高于阈值的样本
            for sample, score in zip(generated_samples, f1_scores):
                if score >= threshold:
                    filtered_samples.append(sample)
        else:
            # 其他语言直接保留
            filtered_samples.extend(generated_samples)

    return filtered_samples


def save_filtered_data(filtered_data, output_file):
    """保存过滤后的数据"""
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for sample in tqdm(filtered_data, desc="Saving filtered data", unit="sample"):  # 添加进度条
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Saved {len(filtered_data)} filtered samples to {output_file}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise


def main(args):
    # 文件路径
    model = args.model.split("/")[-1]
    generated_file_path = f"data/private_syn/{model}/cleaned_private_syndata_dp{args.dp_epsilon}.jsonl"
    original_file_path = f"data/private_syn/{model}/original_data_dp{args.dp_epsilon}.jsonl"
    output_file = f"data/private_syn/{model}/final_private_syndata_dp{args.dp_epsilon}.jsonl"

    # 加载生成数据和原始数据
    with open(generated_file_path, "r") as f:
        generated_data = [json.loads(line) for line in f]

    with open(original_file_path, "r") as f:
        original_data = [json.loads(line) for line in f]

    # 创建问题-语言映射表
    problem_language_map = create_problem_language_mapping(original_data)

    # 按语言分组生成数据
    grouped_generated = group_generated_by_language(generated_data, problem_language_map)
    grouped_original = group_generated_by_language(original_data, problem_language_map)

    # 按相似度过滤数据
    filtered_data = filter_by_similarity(grouped_generated, grouped_original, threshold=0.8)

    # 保存过滤后的数据
    save_filtered_data(filtered_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--dp_epsilon", type=float, default=10.0)
    args = parser.parse_args()

    main(args)
