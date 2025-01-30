python data/private_syn/match_original_data.py \
            --model Qwen/Qwen2.5-Coder-1.5B \
            --dp_epsilon 0.2 \
            --max_lambda 1000 \
            --alpha 0.01 \
            --data_size 55500 \
            --round_trip_model meta-llama/Llama-3.1-70B-Instruct \
            --sim_threshold 0.82 