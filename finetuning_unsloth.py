from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from data_processing import load_system_prompt

import argparse
import sys

parser = argparse.ArgumentParser()

#-m MODEL_NAME -t TASK -i INPUT_DATA -o OUTPUT_MODEL -r RESULTS
parser.add_argument("-m", "--model", help="Model name")
parser.add_argument("-t", "--task", help="Task Name", default="naive")
parser.add_argument("-i", "--input", help="Data input file")
parser.add_argument("-o", "--output", help="Output directory", default="output")



def formatting_prompts_func(examples):
    system_prompt = load_system_prompt(args.task)
    texts = []
    for conversation in examples["conversations"]:
        messages = [
            {"role": "system", "content": system_prompt},
            conversation[0],  # user message
            conversation[1]   # assistant message
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text) 
    return {"text": texts}


def load_model_and_tokenizer(model_name, max_seq_length=2048):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def setup_lora(model):
    lora_params = {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0,
        "bias": "none",
    }
        
    model = FastLanguageModel.get_peft_model(
        model,
        **lora_params,
        use_gradient_checkpointing="unsloth",  
        random_state=3407,
    )
    
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.model or not args.input:
        print("You did not provide a model or data to use.\nPlease add a model to load/run as such: python3 finetuning_unsloth.py -m [MODEL] -i [DATASET].")
        sys.exit(1)
    
    dataset = load_dataset(args.input) # "hoshuhan/amr-3-parsed"
    base_model, tokenizer = load_model_and_tokenizer(args.model)
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    peft_model = setup_lora(base_model)
    
    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        push_to_hub=False,
        num_train_epochs=10,
        warmup_steps=10,
        gradient_accumulation_steps=128,
        logging_steps=20,
        save_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        optim="adamw_8bit"
    )
    
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    
    # Train
    trainer.train()
    
    # Save
    peft_model.save_pretrained(args.output)