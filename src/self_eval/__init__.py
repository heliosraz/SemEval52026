#!/usr/bin/env python3


# Experiment One: Baseline FT
from finetune import SEBaselineFineTune

#TODO: API for configuration file, or, endpoint to integrate w/ main
args = {"lora": {"r": 8,
                 "lora_alpha": 32,
                 "lora_dropout": 0.05,
                 "bias": "none",
                 "task_type": "CAUSAL_LM",
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
                 },
        "train_args": {"output_dir": "results/self_eval",
                       "eval_strategy":"epoch",
                       "push_to_hub": False,
                       "num_train_epochs": 10,
                       "warmup_steps": 10,
                       "gradient_accumulation_steps": 128,
                       "logging_steps": 20,
                       "save_strategy": "epoch",
                       "per_device_train_batch_size": 1,
                       "per_device_eval_batch_size": 1,
                       "optim": "adamw_torch_4bit"},
        "input_fp": "data/",
        "model_out_fp": "checkpoints/",
        "task": "naive",
        "model": "qwen3.0",
        "seq_len": 1024,
        "dtype": "none",
        "4bit": True
        }

baseline_finetune = SEBaselineFineTune(args).run()

# Experiment Two: Self Evaluation head



# Experiment Three: Synonymy
