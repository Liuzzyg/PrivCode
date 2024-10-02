from huggingface_hub import HfApi, HfFolder

# 创建API对象
api = HfApi()

# 上传文件
# pirvate syndata
# api.upload_file(
#     path_or_fileobj="data/private_syn/cleaned_private_syndata.jsonl",  # 本地文件路径
#     path_in_repo="cleaned.jsonl",  # 在repo中的保存路径
#     repo_id='liuzzyg/private-syndata',  # 替换为你的用户名和数据集名称
#     repo_type="dataset"  # 指定repo的类型为数据集
# )

# original data
api.upload_file(
    path_or_fileobj="data/private_syn/original_data.jsonl",  # 本地文件路径
    # path_or_fileobj='data/private_syn/cleaned_private_syndata_test.jsonl',
    path_in_repo="original.jsonl",  # 在repo中的保存路径
    repo_id='liuzzyg/magi-original-data',  # 替换为你的用户名和数据集名称
    repo_type="dataset"  # 指定repo的类型为数据集
)