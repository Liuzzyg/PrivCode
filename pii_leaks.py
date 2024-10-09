import json
import time
import pdb
import argparse

from tqdm import tqdm
from generate.generate import GEN_SOLUTION
from pii_leakage.Prompt import get_all_queries

def query_and_save(queries, base_checkpoint, model_checkpoint, output_file):
    with open(output_file, 'a') as f:
        for query in tqdm(queries):
            prompt = query.prompt.text
            # pdb.set_trace()
            
            try:
                generated_text, subsequences = GEN_SOLUTION(base_checkpoint=base_checkpoint, model_checkpoint=model_checkpoint, prompt=prompt)
            except Exception as e:
                print(f"Error generating for prompt: {prompt}\n{str(e)}")
                generated_text = None
                subsequences = None

            result = {
                'prompt': prompt,
                'generated_text': generated_text,
                'subsequences': subsequences
            }
            f.write(json.dumps(result) + '\n')

    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_checkpoint", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument("--model_checkpoint", type=str, default="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/deepseek-coder-6.7b-base/dp10_syndata/checkpoint-130")
    parser.add_argument("--output_file", type=str, default='pii_leakage/queries/deepseek-coder-6.7b-base_dp10_syndata_step130.jsonl')

    args = parser.parse_args()

    queries = get_all_queries()

    query_and_save(queries, base_checkpoint=args.base_checkpoint, model_checkpoint=args.model_checkpoint, output_file=args.output_file)
