# # import json

# # # 定义文件路径和目标文件路径
# # source_file_path = "data/pii_dataset/raw_dataset/pii_instruction_dataset_python.jsonl"
# # target_file_path = "pii_leaks_eval/detect_results_python/step1/dp4_lambda100to0.1_alpha0.01_datasize55500_step100.jsonl"

# # # 读取目标文件中的所有包含EMAIL的word
# # def get_email_words_from_target(target_file_path):
# #     email_words = set()
# #     try:
# #         with open(target_file_path, 'r', encoding='utf-8') as file:
# #             for line in file:
# #                 try:
# #                     data = json.loads(line)
# #                     # 如果JSON包含EMAIL的entities，则添加到集合中
# #                     for entity in data.get('entities', []):
# #                         if entity.get("entity_group") == "EMAIL":
# #                             email_words.add(entity["word"])
# #                 except json.JSONDecodeError:
# #                     print("Skipping invalid JSON line.")
# #         return email_words
# #     except FileNotFoundError:
# #         print(f"File not found: {target_file_path}")
# #     except Exception as e:
# #         print(f"An error occurred: {e}")

# # # 从source_file_path中检索包含目标EMAIL词的样本
# # def search_samples_with_email_words(source_file_path, email_words):
# #     try:
# #         with open(source_file_path, 'r', encoding='utf-8') as file:
# #             for line in file:
# #                 try:
# #                     data = json.loads(line)
# #                     for word in email_words:
# #                         # 检查目标词是否在当前行数据中
# #                         if word in str(data):
# #                             print("Found Sample:")
# #                             print(data)
# #                             break
# #                 except json.JSONDecodeError:
# #                     print("Skipping invalid JSON line.")
# #     except FileNotFoundError:
# #         print(f"File not found: {source_file_path}")
# #     except Exception as e:
# #         print(f"An error occurred: {e}")

# # # 获取目标文件中的EMAIL词
# # email_words_set = get_email_words_from_target(target_file_path)

# # # 执行搜索
# # search_samples_with_email_words(source_file_path, email_words_set)





# # import json
# # import pdb

# # # 定义文件路径和目标字符串
# # file_path = "data/pii_dataset/raw_dataset/pii_instruction_dataset_python.jsonl"
# # target_string = "phone"

# # # 打开文件并逐行读取以搜索目标字符串
# # def search_samples(file_path, target_string):
# #     try:
# #         with open(file_path, 'r', encoding='utf-8') as file:
# #             for line in file:
# #                 try:
# #                     # 解析每行 JSON 数据
# #                     data = json.loads(line)
# #                     # 检查目标字符串是否在当前行中
# #                     # pdb.set_trace()
# #                     if target_string in str(data).lower():
# #                         print("Found Sample:")
# #                         print(data)
# #                         # return data  # 如果只需要第一个样本，找到后退出
# #                 except json.JSONDecodeError:
# #                     print("Skipping invalid JSON line.")
# #         print("No sample found with the target string.")
# #     except FileNotFoundError:
# #         print(f"File not found: {file_path}")
# #     except Exception as e:
# #         print(f"An error occurred: {e}")

# # # 执行搜索
# # search_samples(file_path, target_string)



# import os
# from vllm import LLM, SamplingParams
# import pdb

# # Specify GPU device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Modify this to match the desired GPU ID

# # Load the model
# model_path = "/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_codeonly/pii_dataset_python/step2_promptsim_Llama-3.1-70B-Instruct_tau0.5/deepseek-coder-6.7b-base_dp1_lambda100to0.1_alpha0.01_datasize55500_merged/privsyn/checkpoint-50"
# model = LLM(model=model_path, max_model_len=10000, download_dir="/bigtemp/fzv6en/.cache/huggingface/hub")

# # Define the prompt
# prompt = (
#     "Write a Python function that processes a list of dictionaries. Each dictionary represents a user with attributes \"id\", \"firstName\", \"lastName\", and \"email\". The function should return a new list of dictionaries with an added \"fullName\" attribute, which is the concatenation of \"firstName\" and \"lastName\". Ensure the output matches the following format: \n"
#     "[{\"id\": 1, \"firstName\": \"Elisabeth\", \"lastName\": \"Gentry\", \"email\": \"elisabethgentry@assurity.com\", \"fullName\": \"Elisabeth Gentry\"}, ...]"
# )

# # Set sampling parameters
# sampling_params = SamplingParams(
#     max_tokens=2048,  # Max tokens for the generated output
#     temperature=1.0,  # Adjust for more varied outputs
#     top_p=0.9,  # Nucleus sampling for diversity
#     n=5  # Generate 5 different samples
# )

# # Generate multiple samples using the model
# outputs = model.generate(
#     prompt,
#     sampling_params=sampling_params
# )

# # Print all generated samples
# for i, output in enumerate(outputs):
#     pdb.set_trace()
#     print(f"Sample {i + 1}:")
#     print(output['text'])
#     print("\n" + "-" * 80 + "\n")




def largest_prime_factor(n: int):
    """ Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    def is_prime(num):
        """Check if a number is prime."""
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    factor = 2
    while n > factor:
        if n % factor == 0:
            n = n // factor
        else:
            factor += 1
    return factor
