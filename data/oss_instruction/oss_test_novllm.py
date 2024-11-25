import json
import torch
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm


def generate_problem_and_solution(model, tokenizer, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return generated_text


def load_model_and_tokenizer(model_name='microsoft/Phi-3-mini-4k-instruct'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('successfully load tokenizer!')
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/bigtemp/fzv6en/.cache/huggingface/hub",)
    model = model.to('cuda')  # Move model to GPU
    return model, tokenizer


def load_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def save_jsonl(output_path, data):
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main(input_path, output_path):
    model, tokenizer = load_model_and_tokenizer()

    data = load_jsonl(input_path)
    results = []

    for sample in tqdm(data, desc="Generating problems and solutions"):
        code_snippet = sample['content']
        prompt = (
            "Please gain inspiration from the following random code snippet related to the Numpy API to create a high-quality programming problem. We need that the created programming problem only uses numpy api, and don't use any other libraries. Present your output in two distinct sections: [Problem Description]: and [Solution]:, and each section end up with [END].\n\n"
            f"Code snippet for inspiration:\n{code_snippet}\n\n"
            "Guidelines for each section:\n\n"
            "1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included. Limit the [Problem Description] to a short paragraph.\n\n"
            "2. [Solution]: Offer a comprehensive, simple, **correct** solution that accurately addresses the [Problem Description] you provided, and use **only the NumPy api**."
        )
        result = generate_problem_and_solution(model, tokenizer, prompt)
            
        pdb.set_trace()
        
        # try:
        #     problem_description, solution = result.split("[Solution]:", 1)
        #     problem_description = problem_description.split("[Problem Description]:", 1)[-1].strip()
        #     solution = solution.strip()
        # except ValueError:
        #     problem_description = result
        #     solution = "Solution not provided."

        results.append({
            'index': sample['index'],
            # 'problem_description': problem_description,
            # 'solution': solution
            'completion': result
        })

    save_jsonl(output_path, results)

input_path = 'data/oss_instruction/random_numpy_code_snippet.jsonl'
output_path = 'data/oss_instruction/oss.jsonl'
main(input_path, output_path)
