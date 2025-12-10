from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, LoftQConfig
import numpy as np
import evaluate

import sys
from tqdm import tqdm

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # convert the logits to their predicted class
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def formatting_prompts_func(examples):
    system_prompt = "You are an AMR parser. Convert English sentences into Abstract Meaning Representation (AMR) graphs. Use proper AMR notation and formatting."
    texts = []
    for conversation in examples["conversations"]:
        messages = [
        {" role ": " system " , " content ": system_prompt } ,
        conversation[0] , # user message
        conversation[1] # assistant message
        ]
        text=tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text) 
    return {"text": texts }


if __name__=="__main__":
    dataset = load_dataset("hoshuhan/amr-3-parsed")
    # print(dataset["conversations"])
    
    lora_config = LoraConfig(
        r=8 , # Rank dimension
        lora_alpha=32 , # Alpha parameter for LoRA scaling
        target_modules=[
            "q_proj", # Query projection
            "k_proj", # Key projection
            "v_proj", # Value projection
            "o_proj" # Output projection
            ],
            lora_dropout=0.05, # Dropout probability
            bias="none", # No bias parameters
            task_type="CAUSAL_LM", # Task type for causal language modeling
        )
    
    base_model = AutoModelForCausalLM.from_pretrained(sys.argv[1])
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    dataset = dataset.map(formatting_prompts_func, batched = True)
    
    peft_model = get_peft_model(base_model, lora_config)

    training_args = TrainingArguments(
        output_dir="amr_parser",
        eval_strategy="epoch",
        push_to_hub=False,
        num_train_epochs=10,
        warmup_steps=10,
        gradient_accumulation_steps=128
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics
    )
    trainer.train()
    
    
    