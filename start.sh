vllm serve Qwen/Qwen3-0.6B --max_num_batched_tokens 37376 \
                        --max_model_len 37376 \
                        --enforce-eager \