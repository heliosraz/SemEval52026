from data_processing import load_data
import torch
from tqdm import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from safetensors.torch import save_file, load_model
import os
import models
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from typing import List, Dict, Tuple, Any

'''
To run script:
python main.py "sentence-transformers/all-roberta-large-v1" data/train.json data/dev.json
'''

class WordSenseData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"average": self.data.loc[idx, "average"],
                "stdev": self.data.loc[idx, "stdev"],
                "index": self.data.loc[idx, 'index'],
                "homonym": self.data.loc[idx, 'homonym'],
                "context": self.data.loc[idx, 'context'],
                "judged_meaning": self.data.loc[idx, "judged_meaning"],
                "example_sentence": self.data.loc[idx, 'example_sentence']}

class MaskedData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"masked": self.data.loc[idx, "masked"],
                "candidate": self.data.loc[idx, 'candidate'],
                "context": self.data.loc[idx, 'context']}

class CrossAttentionData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"average": self.data.loc[idx, "average"],
                "stdev": self.data.loc[idx, "stdev"],
                "candidate": self.data.loc[idx, 'candidate'],
                "context": self.data.loc[idx, "context"]}

os.makedirs("checkpoint", exist_ok = True)
os.environ["TOKENIZERS_PARALLELISM"] = "False"
task_dataset = {'data': WordSenseData, 'classifier': WordSenseData,'finetuning': CrossAttentionData,'pretraining':MaskedData}

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

def train(
        model,
        train_set: Dataset,
        dev_set: Dataset,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        input_tags: str|List[str] = ["context", "judged_meaning"],
        label_tag: str = "label",
        n_epochs: int = 100,
        batch_size=64,
        freeze_schedule: Dict[int,(Tuple[List[torch.Tensor]]|Any)] = {},
        save_weights: bool = True,
        metric = None):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_loader = DataLoader(train_set, 
                        batch_size=batch_size, 
                        shuffle=True,)
    dev_loader = DataLoader(dev_set, 
                        batch_size=batch_size)
    
    train_loss_record = []
    train_acc_record = []
    dev_loss_record = []
    dev_acc_record = []
    
    best_vloss = 1_000_000.
    for epoch in tqdm(range(n_epochs),desc="Epochs:", position = 0):
        if epoch in schedule:
            # Freeze
            for layer in freeze_schedule[epoch][0]:
                for param in layer.parameters():
                    param.require_grad = False
            # Unfreeze
            for layer in freeze_schedule[epoch][1]:
                for param in layer.parameters():
                    param.require_grad = True
        running_tacc = 0
        running_loss = 0.0
        model.train()
        for batch in tqdm(train_loader, desc="Training Batch:", leave = False):
            y_labels = torch.Tensor(batch[label_tag])
            X_batch = batch
            optimizer.zero_grad()
            y_pred = model(X_batch, train = True, select = input_tags)
            loss = loss_fn(y_pred, y_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_tacc += metric(y_pred, y_labels) # accuracy
            
        avg_loss = running_loss/len(train_loader)
        running_vloss = 0.0
        
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_vacc = 0
        with torch.no_grad():
            for v_batch in tqdm(dev_loader, desc="Dev Batch:", leave = False):
                v_labels = torch.Tensor(v_batch[label_tag])
                optimizer.zero_grad()
                v_pred = model(v_batch, train = True, select = input_tags)
                v_loss = loss_fn(v_pred, v_labels)
                
                running_vloss += v_loss.item()
                running_vacc += metric(v_pred, v_labels) # acc

        avg_vloss = running_vloss/len(dev_loader)
        print('LOSS train {} dev {}'.format(avg_loss, avg_vloss))
        print('ACCURACY train {} dev {}'.format(running_tacc/len(train_set), running_vacc/len(dev_set)))
        
        train_loss_record.append(avg_loss)
        train_acc_record.append(running_tacc/len(train_set))
        dev_loss_record.append(avg_vloss)
        dev_acc_record.append(running_vacc/len(dev_set))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            
            best_vloss = avg_vloss
            model_name = "{}_{}_{}".format(
                                        base_model.split("/")[-1],
                                        timestamp,
                                        epoch)
            model_dir = "checkpoint/{}".format(model_name)
            os.makedirs(model_dir, exist_ok = True)
            
            metrics = {
                "train": {
                        "loss": train_loss_record,
                        "acc": train_acc_record},
                "dev": {
                        "loss": dev_loss_record,
                        "acc": dev_acc_record}}
            
            save_model(model, metrics, model_dir, model_name)
            if save_weights:
                plot_linear_weights(
                    [param.data for name, param in model.named_parameters() if "weight" in name or "bias" in name],
                    [name for name, _ in model.named_parameters()],
                    running_tacc/len(train_set),
                    running_vacc/len(dev_set),
                    avg_loss,
                    avg_vloss,
                    "checkpoint/{}/{}.png".format(model_name,model_name)
                    )
            
    return model_path

def run(model, data: pd.DataFrame):
    loader = DataLoader(data, 
                    batch_size= 64,)
    res = pd.DataFrame(columns = ["id", "prediction"])
    with torch.no_grad():
        for batch in loader:
            pred = torch.argmax(model(batch), dim=1)+1
            y = pd.DataFrame(pred.cpu(), columns=['prediction'])
            y['id'] = batch['index']
            res = pd.concat([res, y])
    res["id"] = res["id"].astype("str")
    res["prediction"] = res["prediction"].astype("int")
    return res

def plot_linear_weights(weights_list, layer_names, train_acc, dev_acc, train_loss, dev_loss, save_path='weights_visualization.png', figsize=(18, 5)):
    
    fig, axes = plt.subplots(1, len(weights_list), figsize=figsize)
    
    # Plot each weight matrix
    for weights, name, ax in zip(weights_list, layer_names, axes):
        weights = weights.detach().cpu().numpy()
        
        im = ax.matshow(weights, cmap='coolwarm', aspect='auto', vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
        ax.set_title(f'{name}\nShape: {weights.shape}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Input Features')
        ax.set_ylabel('Output Features')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Create metrics caption
    caption = (
        f'Training: Acc = {train_acc:.4f}, Loss = {train_loss:.4f} | '
        f'Dev: Acc = {dev_acc:.4f}, Loss = {dev_loss:.4f}')
    
    # Add caption below the plots
    fig.text(0.5, 0.02, caption, ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for caption
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.close()
        
def save_model(
                model,
                metrics,
                model_dir = "./checkpoint",
                model_name = "model"):
    model_fpath = os.path.join(model_dir,'{}.safetensors'.format(model_name,model_name))
    model_mpath = os.path.join(model_dir,'{}_metrics.json'.format(model_name,model_name))
    state_dict = model.state_dict()
    save_file(state_dict, model_fpath)
    with open(model_mpath, "w") as f:
        json.dump(metrics, f, indent = 4)

if __name__ == "__main__":
    def accuracy_count(labels, preds):
        return sum([mn<=label<=mx for label, (mn, mx) in zip(labels, preds)])
    ## Data Processing
    if len(argv)<3:
        print("No data files were provided.")
        exit(1)
    elif len(argv)==3:
        base_model = ""
        train_df = load_data(argv[1])
        dev_df = load_data(argv[2])
    else:
        base_model = argv[1]
        train_df = load_data(argv[2])
        dev_df = load_data(argv[3])
    task = argv[2].split('/')[-2]
    train_set = task_dataset[task](train_df)
    dev_set = task_dataset[task](dev_df)

    ## Training config
    if base_model:
        model = models.GeneralistModelScored(base_model).to(device)
    else:
        model = models.GeneralistModelScored().to(device)
    metric = accuracy_count
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    optim = torch.optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': 1e-5, 'weigh_decay': 0.2},
        {'params': model.classifier.parameters(), 'lr': 1e-4, 'weigh_decay': 0.01}
        ],
        betas=(0.7, 0.999))
    schedule = {
        0: ([model.classifier],[]), 
        30: ([model.base_model],[model.classifier]),
        60:([],[model.base_model])}
    
    # Model Training and Eval
    model_path = train(
                    model,
                    train_set,
                    dev_set,
                    label_tag = "average",
                    loss_fn = loss_fn,
                    optimizer = optim,
                    freeze_schedule = schedule,
                    metric = metric,
                    )
    # model_path = "checkpoint/all-mpnet-base-v2_20260109_010623_49.safetensors"
    load_model(model, model_path)
    res = run(model, dev_set)
    
    ## Saving Results
    res.to_json(f"predictions-{base_model.split("/")[-1]}.jsonl",orient="records",lines=True)
