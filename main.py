from data_processing import load_data
import torch
from tqdm import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from safetensors.torch import save_file, load_model
import os
import models
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from typing import List, Dict, Tuple, Any
import wandb
import metrics
from transformers import get_cosine_schedule_with_warmup

'''
To run script:
python main.py "sentence-transformers/all-roberta-large-v1" data/train.json data/dev.json
'''

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
print("DEVICE: {}".format(device))

if device == torch.device("cuda"):
    tensor_parallel = True
else:
    tensor_parallel = False

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
                "source": self.data.loc[idx, 'context'],
                "target": self.data.loc[idx, "judged_meaning"]}
    
class AugWordSenseData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {col: self.data.loc[idx, col] for col in self.data.columns.tolist()}

class CrossAttentionData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"average": self.data.loc[idx, "average"],
                "stdev": self.data.loc[idx, "stdev"],
                "candidate": self.data.loc[idx, 'candidate'],
                "full_context": self.data.loc[idx, "full_context"]}

os.makedirs("checkpoint", exist_ok = True)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

task_dataset = {
    'data': WordSenseData,
    'classifier': CrossAttentionData,
    'finetuning': CrossAttentionData,
    'pretraining':AugWordSenseData
    }
model_key = {
    'GeneralistModel_nosep': models.GeneralistModel_nosep,
    'GeneralistModel': models.GeneralistModel,
    'PretrainedGeneralistModel': models.PretrainedGeneralistModel,
    'PretrainedGeneralistModel': models.PretrainedGeneralistModel
    }
metric_key = {
    "mask": metrics.accuracy,
    "average": metrics.range,
}


def train(
        model,
        train_set: Dataset,
        dev_set: Dataset,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        input_tags: str|List[str] = ["full_context", "judged_meaning"],
        label_tag: str = "label",
        metric_label: str = "label",
        metric = None,
        n_epochs: int = 100,
        batch_size=64,
        freeze_schedule: Dict[int,(Tuple[List[torch.Tensor]]|Any)] = {},
        save_weights_plots: bool = True,
        mask: bool = False,
        k: int = 2):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    top_k = []
    train_loader = DataLoader(train_set, 
                        batch_size=batch_size, 
                        shuffle=True,)
    dev_loader = DataLoader(dev_set, 
                        batch_size=batch_size)
    
    best_vloss = 1_000_000.
    for epoch in tqdm(range(n_epochs),desc="Epochs:", position = 0):
        if epoch in freeze_schedule:
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
            # mask needs to tokenize to know sep_len -> outputs input_ids (w/ mask)
                # returning text with mask, risks the seq_len changing
                # masking changes the task s.t. the y_labels change
            # model deals with tokenizing, so:
                # 1. move masking to model,  returns y_pred and y_labels
                # 2. create a tag to disable model tokenizing
            X_batch = batch
            optimizer.zero_grad()
            y_pred, mask_keys = model(X_batch, select = input_tags, mask = mask)
            batch["mask"] = torch.Tensor(mask_keys["mask_ids"])
            y_labels = torch.stack(batch[label_tag], dim = 1).float() if type(batch[label_tag])==list else batch[label_tag]
            if type(loss_fn)==torch.nn.CrossEntropyLoss:
                y_labels = y_labels.long()
            else:
                y_pred = torch.softmax(y_pred, dim = 1)
            loss = loss_fn(y_pred, y_labels.to(device))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss += loss.item()
            # print(loss.item())
            y_metric = torch.stack(batch[metric_label], dim = 1).float() \
                if type(batch[metric_label])==list else batch[metric_label]
            running_tacc += metric(torch.argmax(y_pred, dim = 1), y_metric.to(device)).item() # accuracy
            
        avg_loss = running_loss/len(train_loader)
        running_vloss = 0.0
        
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_vacc = 0
        with torch.no_grad():
            for v_batch in tqdm(dev_loader, desc="Dev Batch:", leave = False):
                v_pred, mask_keys = model(v_batch, select = input_tags, mask = mask)
                v_batch["mask"] = torch.Tensor(mask_keys["mask_ids"])
                v_labels = torch.stack(v_batch[label_tag], dim = 1).float() \
                    if type(v_batch[label_tag])==list else v_batch[label_tag]
                if type(loss_fn)==torch.nn.CrossEntropyLoss:
                    v_labels = y_labels.long()
                else:
                    v_pred = torch.softmax(v_pred, dim = 1)
                v_loss = loss_fn(v_pred, v_labels.to(device))
                
                running_vloss += v_loss.item()
                v_metric = torch.stack(v_batch[metric_label], dim = 1).float() \
                    if type(v_batch[metric_label])==list else v_batch[metric_label]
                running_vacc += metric(torch.argmax(v_pred, dim = 1), v_metric.to(device)).item() # acc

        avg_vloss = running_vloss/len(dev_loader)
        print('LOSS train {} dev {}'.format(avg_loss, avg_vloss))
        print('ACCURACY train {} dev {}'.format(running_tacc/len(train_set), running_vacc/len(dev_set)))
        
        run.log({"train_loss":avg_loss, "train_acc":running_tacc/len(train_set), "valid_loss":avg_vloss, "valid_acc":running_vacc/len(dev_set)})

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            
            best_vloss = avg_vloss
            model_name = "{}_{}_{}".format(
                                        model.name.split("/")[-1],
                                        timestamp,
                                        epoch)
            model_dir = "checkpoint/{}".format(model_name)
            os.makedirs(model_dir, exist_ok = True)
            state_dict = get_state_dict(model)
            top_k.append(state_dict)
            if len(top_k)>k:
                top_k.pop(0)
            if save_weights_plots:
                plot_linear_weights(
                    [model.base_model.K.weight, model.base_model.Q.weight, model.base_model.V.weight],
                    ["model.base_model.K", "model.base_model.Q", "model.base_model.V"],
                    running_tacc/len(train_set),
                    running_vacc/len(dev_set),
                    avg_loss,
                    avg_vloss,
                    "checkpoint/{}/{}.png".format(model_name,model_name)
                    )
        if avg_vloss>20:
            break
    save_model(top_k, model_dir, model_name)
    return model_dir

def eval(model, data: pd.DataFrame):
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
    run.log({"weights_visualization": wandb.Image(fig)})
    print(f"Figure saved to wandb")
    plt.close('all')

def get_state_dict(model):
    state_dict = {
        name: param 
        for name, param in model.named_parameters() 
        if param.requires_grad
    }
    return state_dict

def save_model(
                stat_dicts,
                model_dir = "./checkpoint",
                model_name = "model"):
    model_fpath = os.path.join(model_dir,'{}.safetensors'.format(model_name,model_name))
    for state_dict in stat_dicts:
        save_file(state_dict, model_fpath)

if __name__ == "__main__":    
    run = wandb.init(
        entity="heliosra-n-a",
        project="2026set5",
        settings=wandb.Settings(
            x_disable_stats=True,
            console="off",
            save_code=False,
            quiet=True,
        )
    )
    config = wandb.config
    
    task = "pretraining"
    train_df = load_data(config.train_data)
    dev_df = load_data(config.dev_data)
    train_set = task_dataset[task](train_df)
    dev_set = task_dataset[task](dev_df)

    ## Training parameters
    base = model_key[config.architecture]
    wrapper = model_key[config.wrapper]
    model_name = config.encoder
    model = wrapper(
        base=base,
        model_name=model_name,
        d_attn=config.d_attn,
        drop_attn = config.drop_attn,
        drop_cls = config.drop_cls).to(device)
    
    input_tags = ["source", "target"]
    label_tag = "mask"
    metric_label = "mask"
    metric = metric_key[metric_label]
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optim_params = {
        "betas": (0.7, 0.999),
        "lr": config.lr_others,
        "weight_decay": config.weight_decay_other
        }
    optim = torch.optim.AdamW([
        {
            'params': model.base_model.model.parameters(),
            'lr': config.lr_encoder_pre,
            'weight_decay': config.weight_decay_encoder
            },
        {
            'params': [param \
                for name, param in model.base_model.named_parameters() \
                    if "K" in name or "Q" in name or "V" in name],
            'lr': config.lr_encoder,
            'weight_decay': config.weight_decay_encoder
            }
        ],
        betas=(0.7, 0.999))
    total_steps = len(train_set)//config.batch_size*config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=total_steps*.10,
        num_training_steps=total_steps*.90
    )
    freeze_schedule = {
        0: ([model.classifier],[])}
    mask = True
    
    # Double checking config feasibility
    if (type(loss_fn), label_tag) == (torch.nn.CrossEntropyLoss, "probs") or \
        (type(loss_fn), label_tag) == (torch.nn.KLDivLoss, "mask") or \
            (type(loss_fn), label_tag) == (torch.nn.KLDivLoss, "average"):
        raise TypeError("Loss function and label tag mismatch. \
                        Please check the training parameters:\n\
                        type(loss_fun), label_tag = {},{}".format(type(loss_fn), label_tag))
    
    # Model Training and Eval
    model_path = train(
                    model,
                    train_set,
                    dev_set,
                    input_tags = input_tags,
                    label_tag = label_tag,
                    metric_label = metric_label,
                    loss_fn = loss_fn,
                    optimizer = optim,
                    lr_scheduler = scheduler,
                    metric = metric,
                    n_epochs = config.epochs,
                    batch_size = config.batch_size,
                    freeze_schedule = freeze_schedule,
                    mask = mask,
                    )
    wandb.finish()
    ################
    
        
    #     ## Data Processing
    # if len(argv)<3:
    #     print("No data files were provided.")
    #     exit(1)
    # elif len(argv)==3:
    #     base_model = ""
    #     train_df = load_data(argv[1])
    #     dev_df = load_data(argv[2])
    # else:
    #     base_model = argv[1]
    #     train_df = load_data(argv[2])
    #     dev_df = load_data(argv[3])
    # task = "pretraining"
    # train_set = task_dataset[task](train_df)
    # dev_set = task_dataset[task](dev_df)

    # ## Training parameters
    # if base_model:
    #     model = models.PretrainedGeneralistModel(
    #         base=models.GeneralistModel,
    #         model_name=base_model).to(device)
    # else:
    #     model = models.PretrainedGeneralistModel(
    #         base=models.GeneralistModel,).to(device)
    # input_tags = ["source", "target"]
    # label_tag = "mask"
    # metric_label = "mask"
    # epochs = 10
    # batch_size = 32
    # metric = metrics.accuracy
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optim_params = {
    #     "betas": (0.7, 0.999),
    #     "lr": 1e-3,
    #     "weight_decay": 0.2
    #     }
    # optim = torch.optim.AdamW([
    #     {
    #         'params': model.base_model.model.parameters(),
    #         'lr': 1e-3,
    #         'weight_decay':0.2
    #         },
    #     {
    #         'params': [param \
    #             for name, param in model.base_model.named_parameters() \
    #                 if "K" in name or "Q" in name or "V" in name],
    #         'lr': 1e-3,
    #         'weight_decay': 0.2
    #         }
    #     ],
    #     betas=(0.7, 0.999))
    # total_steps = len(train_set)//batch_size*epochs
    # scheduler = get_cosine_schedule_with_warmup(
    #     optim,
    #     num_warmup_steps=total_steps*.10,
    #     num_training_steps=total_steps*.90
    # )
    # freeze_schedule = {
    #     0: ([model.classifier],[])}
    # mask = True
    
    # # Model Training and Eval
    # model_path = train(
    #                 model,
    #                 train_set,
    #                 dev_set,
    #                 input_tags = input_tags,
    #                 label_tag = label_tag,
    #                 metric_label = metric_label,
    #                 loss_fn = loss_fn,
    #                 optimizer = optim,
    #                 lr_scheduler = scheduler,
    #                 metric = metric,
    #                 n_epochs = epochs,
    #                 batch_size = batch_size,
    #                 freeze_schedule = freeze_schedule,
    #                 mask = mask,
    #                 )
