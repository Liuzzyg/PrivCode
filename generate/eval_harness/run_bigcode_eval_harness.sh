# # # astdp
# # accelerate launch  bigcode-evaluation-harness/main.py   --model "/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/starcoder2-3b/dp10_lambda1to0.1_alpha0.01_merged/checkpoint-12"  \
# #                                                         --max_length_generation 512   \
# #                                                         --tasks mbpp   \
# #                                                         --n_samples 1   \
# #                                                         --batch_size 1   \
# #                                                         --temperature 0   \
# #                                                         --do_sample False   \
# #                                                         --precision bf16   \
# #                                                         --allow_code_execution   \
# #                                                         --use_auth_token       \


# # dp10 baseline
# # accelerate launch  bigcode-evaluation-harness/main.py   --model "/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/starcoder2-3b/dp10_baseline_merged/checkpoint-12"  \
# #                                                         --max_length_generation 512   \
# #                                                         --tasks humanevalplus   \
# #                                                         --n_samples 1   \
# #                                                         --batch_size 1   \
# #                                                         --temperature 0   \
# #                                                         --do_sample False   \
# #                                                         --precision bf16   \
# #                                                         --allow_code_execution   \
# #                                                         --use_auth_token       \


# # astdp
# accelerate launch  bigcode-evaluation-harness/main.py   --model "bigcode/starcoder2-3b"  \
#                                                         --max_length_generation 512   \
#                                                         --tasks mbpp   \
#                                                         --n_samples 1   \
#                                                         --batch_size 1   \
#                                                         --temperature 0.01   \
#                                                         --do_sample False   \
#                                                         --precision bf16   \
#                                                         --allow_code_execution   \
#                                                         --use_auth_token       \
#                                                         --metric_output_path "generate/eval_harness/starcoder2-3b/pretrain/results_mbpp_test.json" \


# accelerate launch  bigcode-evaluation-harness/main.py   --model "bigcode/starcoder2-3b"  \
#                                                         --max_length_generation 512   \
#                                                         --tasks mbppplus   \
#                                                         --n_samples 1   \
#                                                         --batch_size 1   \
#                                                         --temperature 0.01   \
#                                                         --do_sample False   \
#                                                         --precision bf16   \
#                                                         --allow_code_execution   \
#                                                         --use_auth_token       \
#                                                         --metric_output_path "generate/eval_harness/starcoder2-3b/pretrain/results_mbppplus_test.json" \




# CUDA_VISIBLE_DEVICES=0 accelerate launch  bigcode-evaluation-harness/main.py   --model bigcode/starcoder2-3b   --max_length_generation 512  --tasks mbpp   --n_samples 200   --batch_size 100   --temperature 0.2   --precision bf16   --allow_code_execution   --use_auth_token   --metric_output_path "generate/eval_harness/starcoder2-3b/pretrain/results_mbpp_paperconfig.json"

# CUDA_VISIBLE_DEVICES=1,3 accelerate launch  bigcode-evaluation-harness/main.py   --model bigcode/starcoder2-3b   --max_length_generation 512  --tasks mbppplus   --n_samples 200   --batch_size 100   --temperature 0.2   --precision bf16   --allow_code_execution   --use_auth_token   --metric_output_path "generate/eval_harness/starcoder2-3b/pretrain/results_mbppplus_paperconfig.json"

CUDA_VISIBLE_DEVICES=1 accelerate launch  bigcode-evaluation-harness/main.py   --model /bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/starcoder2-3b/dp10_lambda1to0.1_alpha0.01_merged/checkpoint-15   --max_length_generation 512  --tasks mbpp   --n_samples 200   --batch_size 100   --temperature 0.2   --precision bf16   --allow_code_execution   --use_auth_token   --metric_output_path "generate/eval_harness/starcoder2-3b/astdp/results_mbpp_dp10_lambda1to0.1_alpha0.01_checkpoint-15_paperconfig.json"