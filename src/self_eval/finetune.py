#!/usr/bin/env python3
# SemEval2026 --- Task 5
# OLLAMA implementation of self-evaluation preliminary
# based on sunny's script
# ===========================================================================|
# Import Statements
from datasets import load_dataset
import json
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from data_processing import load_system_prompt
# ===========================================================================|
# Functions

def init_dataset(**params):
    with open(params["input_fp"], 'r') as f:
        return [{"prompt": f"{f['precontext']}{f['sentence']}{f['ending']}",
                 "completion": f["average"]}
                for o in json.load(f)]


def init_model(unsloth=True, **params):
    lora_cfg = LoraConfig(**params["lora"])

    if unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=params["model"],
            max_seq_len=params["seq_len"],
            dtype=params["dtype"],
            load_in_4bit=params["4bit"]
        )
        model = FastLanguageModel.get_peft_model(
            model,
            **params["lora"],
            use_gradient_checkpointing="unsloth",
            random_state=3407  # TODO -> why?
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(params["model"])
        tokenizer = AutoTokenizer.from_pretrained(params["model"])
        model = get_peft_model(model, lora_cfg)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train(dataset, model, tokenizer, **params):
    train_cfg = SFTConfig(**params["train_args"])
    trainer = SFTTrainer(
        model = model,
        args = train_cfg,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"]
    )
    trainer.train()


def run(args):
    model, tokenizer = init_model(args)
    dataset = init_dataset(args)
    train(dataset, model, tokenizer, args)
    model.save_pretrained(args["model_out_fp"])


if __name__ == "__main__":
    args = {"lora": {"r": 8,
                     "lora_alpha": 32,
                     "lora_dropout": 0.05,
                     "bias": "none",
                     "task_type": "CAUSAL_LM",
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
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
            "4bit": True}

    run(args)
