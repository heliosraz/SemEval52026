from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset, DatasetDict

accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits,labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    print(preds.shape)
    return accuracy_metric.compute(predictions=preds, references=labels)

def load_data(path):
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    data['combined'] = data['context'] + '[SEP]' + data['judged_meaning']
    data['label'] = data['average'].astype('float').round(0).astype('int') - 1
    new_data = data.reset_index()
    return new_data[['combined','label']]


# metrics taken from metrics.py in main branch
def accuracy(preds, labels):
    return sum([pred==l for pred, l in zip(preds, labels)])
def range(preds, labels):
    return sum([m[0]<=pred<=m[1] for pred, m in zip(preds, labels)])

#TODO: figure out if can parameterize which features to use (def'n, sentence etc)
def tokenize_data(data):
    return tokenizer(data['combined'], padding="max_length", truncation=True)

def train(model_name:str, data, out_file:str='out.txt', device="cpu"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)
    training_args = TrainingArguments(
            output_dir = 'bert',
            eval_strategy = 'epoch',
            label_names= ['labels']
    )
    loss_func = torch.nn.CrossEntropyLoss()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= data['train'],
        eval_dataset = data['eval'],
        compute_metrics = compute_metrics,
        #compute_loss_func = loss_func
    )
    trainer.train()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name,num_labels=5)
    data = DatasetDict()
    #TODO: cleaner version of this data collating-note had to change load_data 
    
    data['train'] = Dataset.from_pandas(load_data("train.json"))
    data['eval'] = Dataset.from_pandas(load_data("dev.json"))
    tok_data = data.map(tokenize_data, batched=True)
    print(tok_data['train'][0])
    train(model_name= model_name, data=tok_data, device=device)

