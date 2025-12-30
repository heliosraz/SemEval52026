from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

import sys

def formatting_prompts_func(examples):
    system_prompt = "You are an AMR parser. Convert English sentences into Abstract Meaning Representation (AMR) graphs. Use proper AMR notation and formatting."
    for i, conversation in enumerate(examples["conversations"]):
        messages = [
        {"role": "system" , "content": system_prompt } ,
        conversation[0] , # user message
        conversation[1] # assistant message
        ]
        examples["conversation"][i]=tokenizer.apply_chat_template(messages, tokenize=False)


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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset.map(formatting_prompts_func, batched = True)
    
    peft_model = get_peft_model(base_model, lora_config)

    training_args = TrainingArguments(
        output_dir="amr_parser",
        eval_strategy="epoch",
        push_to_hub=False,
        num_train_epochs=10,
        warmup_steps=10,
        gradient_accumulation_steps=128,
        logging_steps=20,
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        optim="adamw_torch_4bit"
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    trainer.train()
    
    
    