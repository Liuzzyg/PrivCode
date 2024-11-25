CUDA_VISIBLE_DEVICES=0,1,2,3 python data/private_syn/generate.py \
                                --model Qwen/Qwen2.5-Coder-1.5B \
                                --ckpt /bigtemp/fzv6en/liuzheng/dpcode/checkpoints_code/magicoder/Qwen2.5-Coder-1.5B/dp10_lambda1000to0.1_alpha0.01_merged/checkpoint-15
