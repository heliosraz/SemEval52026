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

# ===========================================================================|
# Functions
class SEBaselineFineTune:
    def __init__(self, **params):
        self.model, self.tokenizer = self.init_model(params)
        self.dataset = self.init_dataset(params)
        self.params = params

    def run(self):
        self.train(self.params)
        self.model.save(self.params["model_out_fp"])

    def train(self, **params):
        train_cfg = SFTConfig(**params["train_args"])
        trainer = SFTTrainer(
            model=self.model,
            args=train_cfg,
            processing_class=self.tokenizer,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["val"]
        )
        trainer.train()

    def init_dataset(self, **params):
        with open(params["input_fp"], 'r') as f:
            return [{"prompt": f"{f['precontext']}{f['sentence']}{f['ending']}",
                     "completion": f["average"]}
                    for o in json.load(f)]

    def init_model(self, unsloth=True, **params):
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
