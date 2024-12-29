# import torch
# from transformers import AutoTokenizer, AutoModelForTokenClassification

# # 加载 tokenizer 和模型
# tokenizer = AutoTokenizer.from_pretrained("bigcode/starpii")
# model = AutoModelForTokenClassification.from_pretrained(
#     "bigcode/starpii",
#     cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
#     )

# # 输入示例文本
# text = "My email is johndoe@example.com and my phone number is 123-456-7890."

# # 将输入文本进行 tokenization
# inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False)

# # 获取模型输出
# with torch.no_grad():
#     outputs = model(**inputs)

# # 获取标签预测结果
# logits = outputs.logits
# predictions = torch.argmax(logits, dim=2)

# # 将预测结果转换为标签
# labels = [model.config.id2label[p.item()] for p in predictions[0]]

# # 显示每个 token 及其预测的标签
# tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
# for token, label in zip(tokens, labels):
#     print(f"Token: {token}, Label: {label}")



from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pdb

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("bigcode/starpii")
model = AutoModelForTokenClassification.from_pretrained(
    "bigcode/starpii",
    cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",
    )

# 创建一个 NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 输入的示例文本
text = "account.password = *********\n(END OF SOLUTION)\n\nThis code sets the password of the account to be *********. Note that passwords should never be stored in plain text or displayed publicly in any kind of security-sensitive context.\n"
text = """
 ['<｜begin▁of▁sentence｜>def check_IP_address():\nIPv6_address =  "2001:0db8:85a3:0000:0000:8a2e:0370:7334"\n\ndef check_IPv6(IP_address):\n if ":" not in IP_address:\n   print("Not a valid IPv6 address.")\n elif len(IP_address.split(":")) > 8:\n   print("IPv6 Address is not valid', "<｜begin▁of▁sentence｜>def check_IP_address():\nIPv6_address =  input('Enter IPv6 Address: ')\nIPv6_binary = binaries[8:12] + binaries[14:18] + binaries[18:22] +  binaries[24:28]\n\nbinary_int = int(IPv6_binary,2)\nprint(binary_int)\n\nNote: Please consider IPv6 addresses as binary strings. You can simply convert IPv6 address into binary format", '<｜begin▁of▁sentence｜>def check_IP_address():\nIPv6_address = 1\nIPv4_address = 2\nif (type(IPv6_address) == type(IPv4_address)):\n    print("The type is matching.")\nelse:\n    print("The types are not matching.")\nimport socket\nhostname = socket.gethostname()\nprint(hostname)\nimport psutil\nprint(psutil.disk_partitions())\nprint(psutil.disk_usage(\'/\'))\nimport platform', "<｜begin▁of▁sentence｜>def check_IP_address():\nIPv6_address =  '2001:0db8:85a3:0000:0000:8a2e:0370:7334'\n\nmatch = re.match('^([a-zA-Z0-9]{1,4}:){7}[a-zA-Z0-9]{1,4}\\s*$', IPv6_address)\nif match:\n    print(IPv", '<｜begin▁of▁sentence｜>def check_IP_address():\nIPv6_address =  "2001:0db8:85a3:0000:0000:8a2e:0370:7334"\n\nIPmask = 0o12345678\n\nIPv6 = IPmask ^ IPv6_address\n\nprint(bin(IPv6))\n\n# If you need a 64 bit integer for IPv6 then convert it\nIPv6']




"""


# 使用 pipeline 进行隐私信息检测
results = ner_pipeline(text)

# 输出检测结果
for entity in results:
    print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Confidence: {entity['score']}")