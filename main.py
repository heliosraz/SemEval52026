from transformers import AutoTokenizer, AutoModel
from load_data import load_data
import torch
from tqdm import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
from safetensors.torch import save_file, load_model
import os
import models

'''
To run script:
python main.py "sentence-transformers/all-roberta-large-v1" data/train.json data/dev.json
'''

os.makedirs("checkpoint", exist_ok = True)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

from typing import List, Dict

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
    
        
class WordSenseData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
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
        

def train(model, train_set: Dataset, dev_set: Dataset, n_epochs: int = 100, batch_size=64):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # train_set = Subset(train_set, range(200))
    train_loader = DataLoader(train_set, 
                        batch_size=batch_size, 
                        shuffle=True,)
    dev_loader = DataLoader(dev_set, 
                        batch_size=batch_size)
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW([
        {'params': model.K.parameters(), 'lr': 0.004, 'weigh_decay': 0.2},
        {'params': model.Q.parameters(), 'lr': 0.004, 'weigh_decay': 0.2},
        {'params': model.V.parameters(), 'lr': 0.004, 'weigh_decay': 0.2},
        {'params': model.scorer.parameters(), 'lr': 0.01, 'weigh_decay': 0.01}
        ],
        betas=(0.7, 0.999))
    
    best_vloss = 1_000_000.
    for epoch in tqdm(range(n_epochs)):
        if epoch<=10:
            for layer in [model.K, model.Q, model.V]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for layer in [model.K, model.Q, model.V]:
                for param in layer.parameters():
                    param.requires_grad = True
        running_tacc = 0
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            y_batch = torch.Tensor(batch["average"])-1
            y_stdev = torch.Tensor(batch["stdev"])
            y_stdev = y_stdev.masked_fill(y_stdev==0,1e-20)
            
            y_probs = torch.exp(-0.5*((torch.arange(5).unsqueeze(0)-y_batch.unsqueeze(1))/y_stdev.unsqueeze(1))**2).float().to(device)
            y_probs = y_probs / (y_probs.sum(dim=1, keepdim=True) + 1e-8)
            
            X_batch = batch
            optimizer.zero_grad()
            y_pred = model(X_batch, train = True)
            loss = loss_fn(y_pred, y_probs)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            running_tacc += sum([b-std<=a<=b+std for a, b, std in zip(
                            torch.argmax(
                                y_pred,
                                dim = 1).flatten().float().tolist(),
                            y_batch.flatten().float().tolist(),
                            y_stdev.float().tolist())])
            
        avg_loss = running_loss/len(train_loader)
        running_vloss = 0.0
        
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_vacc = 0
        with torch.no_grad():
            for v_data in dev_loader:
                v_batch = torch.Tensor(v_data["average"])-1
                v_stdev = torch.Tensor(v_data["stdev"])
                v_stdev = v_stdev.masked_fill(v_stdev==0,1e-20)
                
                v_probs = torch.exp(-0.5*((torch.arange(5).unsqueeze(0)-v_batch.unsqueeze(1))/v_stdev.unsqueeze(1))**2).float().to(device)
                v_probs = v_probs / (v_probs.sum(dim=1, keepdim=True) + 1e-8)
                
                v_inputs = v_data
                v_outputs = model(v_inputs)
                v_loss = loss_fn(v_outputs, v_probs)
                running_vacc += sum([b-std<=a<=b+std for a, b, std in zip(
                    torch.argmax(v_outputs, dim = 1).flatten().float().tolist(),
                    v_batch.flatten().float().tolist(),
                    v_stdev.float().tolist())])
                running_vloss += v_loss

        avg_vloss = running_vloss/len(dev_loader)
        print('LOSS train {} dev {}'.format(avg_loss, avg_vloss))
        print('ACCURACY train {} dev {}'.format(running_tacc/len(train_set), running_vacc/len(dev_set)))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'checkpoint/{}_{}_{}.safetensors'.format(
                                        base_model.split("/")[-1],
                                        timestamp,
                                        epoch)
            save_model(model, model_path)
            
    return model_path

def run(model, data: pd.DataFrame):
    loader = DataLoader(data, 
                    batch_size= 64,)
    res = pd.DataFrame(columns = ["id", "prediction"])
    with torch.no_grad():
        for batch in loader:
            pred = torch.argmax(model(batch), dim=2)+1
            y = pd.DataFrame(pred.cpu(), columns=['prediction'])
            y['id'] = batch['index']
            res = pd.concat([res, y])
    res["id"] = res["id"].astype("str")
    res["prediction"] = res["prediction"].astype("int")
    return res
        
def save_model(model, model_path="model.safetensors"):
    state_dict = model.state_dict()
    save_file(state_dict, model_path)

if __name__ == "__main__":
    ## Data Processing
    if len(argv)<3:
        print("No data files were provided.")
        exit(1)
    elif len(argv)==3:
        base_model = ""
        train_set = load_data(argv[1])
        dev_set = load_data(argv[2])
    else:
        base_model = argv[1]
        train_set = load_data(argv[2])
        dev_set = load_data(argv[3])
    train_set = WordSenseData(train_set)
    dev_set = WordSenseData(dev_set)

    ## Model Running
    if base_model:
    # model = models.SimilarityScoreModule().to(device)
    # model = models.CrossContentSimilarityModule(base_model).to(device)
        model = models.GeneralistModel(base_model).to(device)
    else:
        model = models.GeneralistModel().to(device)
    # model = SimilarityModule()
    model_path = train(model, train_set, dev_set)
    # model_path = "checkpoint/all-mpnet-base-v2_20260109_010623_49.safetensors"
    load_model(model, model_path)
    res = run(model, dev_set)
    
    ## Saving Results
    res.to_json(f"predictions-{base_model.split("/")[-1]}.jsonl",orient="records",lines=True)
