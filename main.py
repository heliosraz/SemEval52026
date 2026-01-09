from transformers import AutoTokenizer, AutoModel
from load_data import load_data
import torch
from tqdm import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset
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
        

def train(model, train_set: Dataset, dev_set: Dataset, n_epochs: int = 100, batch_size = 8):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_loader = DataLoader(train_set, 
                        batch_size=batch_size, 
                        shuffle=True,)
    dev_loader = DataLoader(dev_set, 
                        batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    best_vloss = 1_000_000.
    best_vacc = 0.
    for epoch in tqdm(range(n_epochs)):
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            y_batch = batch.pop("average")
            y_batch = torch.Tensor(y_batch)-1
            y_batch = y_batch.to(device,
                                dtype=torch.long)
            X_batch = batch
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.to(device, dtype=torch.long))
            # print(f"loss: {loss.item()}")
            # print(f"loss requires_grad: {loss.requires_grad}")
            loss.backward()
            # print(f"y_pred shape: {y_pred.shape}")
            # print(f"y_pred sample: {y_pred[:3]}")  # First 3 predictions
            # print(f"y_pred requires_grad: {y_pred.requires_grad}")

            # print(f"y_batch shape: {y_batch.shape}")
            # print(f"y_batch sample: {y_batch[:3]}")  # First 3 labels
            # print(f"y_batch min/max: {y_batch.min()}/{y_batch.max()}")

            optimizer.step()
            running_loss += loss.item()
        
        # print("=== Gradient Check ===")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: grad is None? {param.grad is None}")
        #         if param.grad is not None:
        #             print(f"  grad mean: {param.grad.mean()}, grad std: {param.grad.std()}")
        avg_loss = running_loss/len(train_loader)
        running_vloss = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        correct = 0
        with torch.no_grad():
            for vdata in dev_loader:
                vlabels = vdata["average"].to(device,
                                            dtype=torch.long)
                vrange = vdata["stdev"]
                vlabels = torch.Tensor(vlabels)-1
                vinputs = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                correct += sum([b-std<=a<=b+std for a, b, std in zip(torch.argmax(voutputs, dim = 1).float().tolist(), vlabels.float().tolist(), vrange.float().tolist())])
                running_vloss += vloss

        avg_vloss = running_vloss/len(dev_loader)
        vacc = correct/len(dev_set)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACCURACY valid {}'.format(vacc))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'checkpoint/{}_{}_{}.safetensors'.format(base_model.split("/")[-1], timestamp,
                                                                epoch)
            save_model(model, model_path)
            
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
        
def save_model(model, model_path="model.safetensors"):
    state_dict = model.state_dict()
    save_file(state_dict, model_path)

if __name__ == "__main__":
    ## Data Processing
    if len(argv)<2:
        print("No data file or model was provided.")
        exit(1)
    else:
        base_model = argv[1]
        train_set = load_data(argv[2])
        dev_set = load_data(argv[3])
    train_set = WordSenseData(train_set)
    dev_set = WordSenseData(dev_set)

    ## Model Running
    # model = models.SimilarityScoreModule().to(device)
    model = models.CrossContentSimilarityModule(base_model).to(device)
    # model = SimilarityModule()
    model_path = train(model, train_set, dev_set)
    # model_path = "models/ambirt_20260108_133030_0.safetensors"
    load_model(model, model_path)
    res = run(model, dev_set)
    
    ## Saving Results
    res.to_json(f"predictions-{base_model.split("/")[-1]}.jsonl",orient="records",lines=True)
