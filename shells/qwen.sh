vllm serve Qwen/Qwen3-0.6B --max_num_batched_tokens 7376 \
                        --max_model_len 7376 \
                        --enforce-eager \
                        --structured-outputs-config.backend guidance \
                        --dtype float32