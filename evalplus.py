import sys
import os
import argparse
import pdb
from typing import List, Dict

from evalplus_copy.humaneval import get_human_eval_plus
from evalplus_copy.utils import write_jsonl

from generate.generate import GEN_SOLUTION_deepseek, batch_GEN_SOLUTION
# from generate.generate_codeqwen import GEN_SOLUTION_CodeQwen
# from generate.generate_codet5 import GEN_SOLUTION_Codet5p
# from generate.generate_phi import GEN_SOLUTION_phi
# from generate.generate_starcoder import GEN_SOLUTION_starcoder



def main(args):
    problems = get_human_eval_plus()
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    # prompts = prompts[:5]
    # task_ids = task_ids[:5]
    # pdb.set_trace()

    solutions = batch_GEN_SOLUTION(
        base_checkpoint=args.checkpoint, 
        batch_size=args.batch_size, 
        prompts=prompts,
        checkpoint_path=args.checkpoint_path,
        is_pretrained=args.is_pretrained,
        is_post_step=args.is_post_step
    )
    
    samples = [{'task_id': task_id, 'solution': solution} for task_id, solution in zip(task_ids, solutions)]

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_jsonl(args.output_path, samples)
    
    # write_jsonl(f"generate/evalplus/magicoder/step1/samples_{model}_dp{args.dp_epsilon}_lbs2024_step{args.step}.jsonl", samples)
    # write_jsonl(f"generate/evalplus/magicoder/step2/samples_{model}_dp{args.dp_epsilon}_test_step{args.step}.jsonl", samples)
    # write_jsonl(f"generate/evalplus/magicoder/step2/samples_{model}_pretrained.jsonl", samples)

    # write_jsonl(f"generate/evalplus/magicoder/samples_{model}_dp{args.dp_epsilon}_step{args.step}.jsonl", samples)
    # write_jsonl(f"generate/evalplus/magicoder/samples_{model}_nodp_step{args.step}.jsonl", samples)
    # write_jsonl(f"generate/evalplus/magicoder/samples_{model}_pretrained.jsonl", samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument("--checkpoint_path", type=str, default="examples/starcoder/finetune/checkpoints/CodeQwen1.5-7B/dp_epsilon/final_checkpoint")
    parser.add_argument("--output_path", type=str, default="generate/evalplus/magicoder/step2/samples_model_args.dp_epsilon_test_args.step.jsonl")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--is_pretrained', action="store_true")
    parser.add_argument('--is_post_step', action="store_true")

    args = parser.parse_args()

    main(args)