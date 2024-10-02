import sys
import os
import argparse
import pdb
from typing import List, Dict

from evalplus.data import get_human_eval_plus, write_jsonl
from generate.generate_codeqwen import GEN_SOLUTION_CodeQwen
from generate.generate_codet5 import GEN_SOLUTION_Codet5p
from generate.generate import GEN_SOLUTION_deepseek, batch_GEN_SOLUTION_deepseek
from generate.generate_phi import GEN_SOLUTION_phi
from generate.generate_starcoder import GEN_SOLUTION_starcoder





def process_problems(problems: Dict[str, Dict[str, str]], checkpoint: str, dp_epsilon: str, epoch: str, step, batch_size) -> None:

    # 从问题中提取提示
    prompts = [problem["prompt"] for problem in problems.values()]
    task_ids = list(problems.keys())

    # prompts = prompts[:5]
    # task_ids = task_ids[:5]
    # pdb.set_trace()

    solutions = batch_GEN_SOLUTION_deepseek(base_checkpoint=checkpoint, dp_epsilon=dp_epsilon, epoch=epoch, step=step, batch_size=batch_size, prompts=prompts)
    
    samples = [{'task_id': task_id, 'solution': solution} for task_id, solution in zip(task_ids, solutions)]

    # write_jsonl(f"generate/evalplus/alpaca_test/samples_{model}_dp{args.dp_epsilon}_epoch{args.epoch}.jsonl", samples)
    
    write_jsonl(f"generate/evalplus/alpaca_step/samples_{model}_dp{args.dp_epsilon}_step{args.step}.jsonl", samples)
    # write_jsonl(f"generate/evalplus/synthetic_numpy/samples_{model}_nodp_epoch{args.epoch}.jsonl", samples)
    # write_jsonl(f"generate/evalplus/alpaca_step/samples_{model}_pretrained.jsonl", samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="deepseek-ai/deepseek-coder-1.3b-base", help="which results to run")
    # parser.add_argument("--checkpoint", type=str, default="bigcode/starcoder2-3b", help="which results to run")
    parser.add_argument('--dp_epsilon', type=str, default='10', help="the epsilon value for differential privacy")
    parser.add_argument('--epoch', type=str, default='19', help="the epsilon value for differential privacy")
    parser.add_argument('--batch_size', type=int, default=32, help="the epsilon value for differential privacy")
    parser.add_argument('--step', type=int, default=32, help="the epsilon value for differential privacy")

    args = parser.parse_args()
    
    # pdb.set_trace()
    # checkpoint = "Qwen/CodeQwen1.5-7B"
    # checkpoint = "Salesforce/codet5p-2b"
    # checkpoint = "deepseek-ai/deepseek-coder-6.7b-instruct"
    # checkpoint = "microsoft/Phi-3.5-mini-instruct"

    model = args.checkpoint.split("/")[-1]

    problems = get_human_eval_plus()
    process_problems(problems, checkpoint=args.checkpoint, dp_epsilon=args.dp_epsilon, epoch=args.epoch, step=args.step, batch_size=args.batch_size)

    # samples = []
    # if 'codet5p' in model:
    #     for task_id, problem in get_human_eval_plus().items():
    #         # solution = GEN_SOLUTION_CodeQwen(base_checkpoint=checkpoint, dp_epsilon=dp_epsilon, prompt=problem["prompt"])
    #         solution = GEN_SOLUTION_Codet5p(base_checkpoint=args.checkpoint, dp_epsilon=args.dp_epsilon, prompt=problem["prompt"])
    #         if '770m' or '220m' in model:
    #             samples.append(dict(task_id=task_id, solution=problem["prompt"]+solution))
    #         elif '2b' in model:
    #             samples.append(dict(task_id=task_id, solution=solution))
    #         # pdb.set_trace()
    #         write_jsonl(f"generate/evalplus/alpaca_new/samples_{model}_dp{args.dp_epsilon}.jsonl", samples)
    #         # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_nodp.jsonl", samples)

    # elif 'deepseek' in model:
    #     for task_id, problem in get_human_eval_plus().items():
    #         pdb.set_trace()
    #         solution = GEN_SOLUTION_deepseek(base_checkpoint=args.checkpoint, dp_epsilon=args.dp_epsilon, prompt=problem["prompt"])
    #         samples.append(dict(task_id=task_id, solution=solution))
    #         # pdb.set_trace()
    #         write_jsonl(f"generate/evalplus/alpaca_new/samples_{model}_dp{args.dp_epsilon}.jsonl", samples)
    #         # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_nodp.jsonl", samples)

    # elif 'Phi' in model:
    #     for task_id, problem in get_human_eval_plus().items():
    #         solution = GEN_SOLUTION_phi(base_checkpoint=args.checkpoint, dp_epsilon=args.dp_epsilon, prompt=problem["prompt"])
    #         samples.append(dict(task_id=task_id, solution=solution))
    #         # pdb.set_trace()
    #         # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_dp{dp_epsilon}_epoch4.jsonl", samples)
    #         write_jsonl(f"generate/evalplus/alpaca_new/samples_{model}_new_dp{args.dp_epsilon}_epoch4.jsonl", samples)
    #         # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_nodp.jsonl", samples)

    # elif 'starcoder' in model:
    #     for task_id, problem in get_human_eval_plus().items():
    #         solution = GEN_SOLUTION_starcoder(base_checkpoint=args.checkpoint, dp_epsilon=args.dp_epsilon, prompt=problem["prompt"])
    #         samples.append(dict(task_id=task_id, solution=solution))
    #         # pdb.set_trace()
    #         write_jsonl(f"generate/evalplus/alpaca_new/samples_{model}_dp{args.dp_epsilon}.jsonl", samples)
    #         # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_c_attn_dp{dp_epsilon}.jsonl", samples)
    #         # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_nodp.jsonl", samples)






    # # write_jsonl(f"evalplus/evalplus/generate/samples_{checkpoint}_dp{dp_epsilon}.jsonl", samples)
    # # write_jsonl(f"evalplus/evalplus/generate/samples_{model}_dp{dp_epsilon}.jsonl", samples)