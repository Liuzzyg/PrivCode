export CUDA_VISIBLE_DEVICES=0,1




## step2

MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-base"
MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

STEPS=80
BATCH_SIZE=2
TARGET_EPSILON=10



# evaluate model finetuned on private syndata
CHECKPOINT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/${MODEL_NAME}/dp${TARGET_EPSILON}_syndata_merged/checkpoint-${STEPS}"
SAVE_ROOT_PATH="generate/bigcodebench/magicoder/filtered_dp${TARGET_EPSILON}_syndata"

bigcodebench.generate \
    --model $CHECKPOINT \
    --model_name $MODEL_NAME \
    --split complete \
    --subset full \
    --greedy \
    --bs $BATCH_SIZE \
    --resume \
    --backend vllm \
    --save_root_path $SAVE_ROOT_PATH \
    --step $STEPS \


# # evaluate model finetuned on original data
# CHECKPOINT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step2/magicoder_syndata/${MODEL_NAME}/original_data_merged/checkpoint-${STEPS}"
# SAVE_ROOT_PATH="generate/bigcodebench/magicoder/original_data"

# bigcodebench.generate \
#     --model $CHECKPOINT \
#     --model_name $MODEL_NAME \
#     --split complete \
#     --subset full \
#     --greedy \
#     --bs $BATCH_SIZE \
#     --resume \
#     --backend vllm \
#     --save_root_path $SAVE_ROOT_PATH \
#     --step $STEPS \


# # evaluate pretraining model
# SAVE_ROOT_PATH="generate/bigcodebench/magicoder/pretrain"

# bigcodebench.generate \
#     --model $MODEL_PATH \
#     --model_name $MODEL_NAME \
#     --split complete \
#     --subset full \
#     --greedy \
#     --bs $BATCH_SIZE \
#     --resume \
#     --backend vllm \
#     --save_root_path $SAVE_ROOT_PATH \
#     --step 0 \













# ## step1

# MODEL_PATH="deepseek-ai/deepseek-coder-1.3b-instruct"
# MODEL_NAME=$(echo $MODEL_PATH | awk -F '/' '{print $NF}')

# STEPS=50
# BATCH_SIZE=2
# TARGET_EPSILON=10




# # evaluate model finetuned with dpsgd
# CHECKPOINT="/bigtemp/fzv6en/liuzheng/dpcode/checkpoints_step1/magicoder/${MODEL_NAME}/dp10_lbs256/checkpoint-${STEPS}"
# SAVE_ROOT_PATH="generate/bigcodebench/magicoder/dp${TARGET_EPSILON}_step1"

# bigcodebench.generate \
#     --model $CHECKPOINT \
#     --model_name $MODEL_NAME \
#     --split complete \
#     --subset full \
#     --greedy \
#     --bs $BATCH_SIZE \
#     --resume \
#     --backend vllm \
#     --save_root_path $SAVE_ROOT_PATH \
#     --step $STEPS \



# # # evaluate pretraining model
# # SAVE_ROOT_PATH="generate/bigcodebench/magicoder/pretrain"

# # bigcodebench.generate \
# #     --model $MODEL_PATH \
# #     --model_name $MODEL_NAME \
# #     --split complete \
# #     --subset full \
# #     --greedy \
# #     --bs $BATCH_SIZE \
# #     --resume \
# #     --backend vllm \
# #     --save_root_path $SAVE_ROOT_PATH \
# #     --step 0 \