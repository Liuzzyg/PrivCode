# evalplus.codegen --model "bigcode/starcoder2-7b"  \
#                  --root "generate/evalplus_0.3.1/starcoder2-7b/pretrain" \
#                  --dataset humaneval                    \
#                  --backend vllm                         \
#                  --tp 1 \
#                  --temperature 0.2 \
#                  --n_samples 200 \


evalplus.evaluate --dataset humaneval --samples "generate/evalplus_0.3.1/starcoder2-7b/pretrain/humaneval/bigcode--starcoder2-7b_vllm_temp_0.2.jsonl"