import json
import pdb
from datasets import load_dataset

def process_dataset():
    dataset = load_dataset(
        'terryyz/pii',
        use_auth_token=True,
        cache_dir='/bigtemp/fzv6en/.cache/huggingface/datasets'
    )

    processed_prompts = []
    
    def process_sample(sample):
        # pdb.set_trace()
        fragments = sample['fragments']
        text = sample['text']
        
        if len(fragments) != 0:
            prompt = f"PII_num: {len(fragments)}\n"
            for i, frag in enumerate(fragments, 1):
                prompt += f"PII_category{i}: {frag['category']}\n"
                prompt += f"PII_value{i}: {frag['value']}\n"
            
            prompt += f"\nText: {text}"
            return prompt

    for sample in dataset['test']:
        processed_prompt = process_sample(sample)
        if processed_prompt is not None:
            processed_prompts.append(processed_prompt)
    pdb.set_trace()
    
    output_path = 'pii_leaks_eval/pii_dataset/pii_dataset.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt in processed_prompts:
            json.dump({"text": prompt}, f, ensure_ascii=False)
            f.write('\n')

    print(f"Processed dataset saved to {output_path}")

process_dataset()
