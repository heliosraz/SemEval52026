from transformers import AutoTokenizer, AutoModel
from load_data import load_data
import torch
from tqdm import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from safetensors.torch import save_file, load_model
import os

os.makedirs("models", exist_ok = True)

from typing import List, Dict

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
    
def tokenize(tokenizer, data):
    return tokenizer(list(data), return_tensors='pt', padding=True, return_offsets_mapping=True)

def get_all_offsets(data:Dict, **kwargs):
    res = {}
    # for each offset type
    # name (e.g. "example_sentence") and tok_outs (i.e. tokenizer outputs)
    for name, tok_outs in kwargs.items():
        res[name] = []
        offsets = tok_outs.pop('offset_mapping')
        # for each data instance
        for target, instance, instance_offsets in zip(data['homonym'], data[name], offsets):
            res[name].append(find_offset(target, 
                                        text=instance, 
                                        offsets=instance_offsets))
    return res.values()

def find_offset(target:str, text:str, offsets:list):
    # locate start and end of word in text
    word_loc = text.find(target)
    word_offset = [word_loc, word_loc + len(target)]
    # do binary search to find token offsets containing end of word
    return binary_search(offsets, word_offset[1])

def binary_search(offsets, value):
    low = 0
    high = len(offsets) - 1
    while low <= high:
        mid = low + (high - low)//2
        # offsets has a start and end, make sure its bigger than end
        if offsets[mid][1] < value:
            low = mid + 1
        elif offsets[mid][0] > value:
            high = mid - 1
        else:
            return mid
    return -1

# select target embedding from final tensor outputs
def obtain_final_embeddings(model_outputs, offsets):
    batch_indices = torch.arange(model_outputs.last_hidden_state.size(0))
    return model_outputs.last_hidden_state[batch_indices, offsets, :]

class ScoreModule(torch.nn.Module):

    def __init__(self, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            self.linear = [torch.nn.Linear(1, hidden_sizes[0]).to(device)]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                self.linear.append(torch.nn.Linear(h_i, h_j).to(device))
            self.linear.append(torch.nn.Linear(hidden_sizes[-1], 5).to(device))
        else:
            self.linear = [torch.nn.Linear(1, 5).to(device)]
        self.activation = torch.nn.ReLU().to(device)
        

    def forward(self, x):
        for layer in self.linear[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.linear[-1](x)
        return x

class SimilarityModule(torch.nn.Module):
    # torch no grad should 

    def __init__(self):
        super().__init__()
        self.bert_layer = AutoModel.from_pretrained("google-bert/bert-base-cased").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        self.cosine = torch.nn.CosineSimilarity(dim=1)

    def forward(self, data: Dict):
        # taken from benchmark_bert.py
        context_toks = self.tokenizer(list(data['context']),
                                    return_tensors='pt', 
                                    padding=True,
                                    return_offsets_mapping=True).to(device)
        example_toks = self.tokenizer(list(data['example_sentence']),
                                    return_tensors='pt', 
                                    padding=True, 
                                    return_offsets_mapping=True).to(device)
        # get list of target offsets for each data instance for both context and example
        context_offsets, example_offsets = get_all_offsets(data,
                                                        context=context_toks,
                                                        example_sentence=example_toks)
        context_outputs = self.bert_layer(**context_toks)
        example_outputs = self.bert_layer(**example_toks)
        sim = []
        context_tensor = context_outputs.last_hidden_state[torch.arange(0, context_outputs.last_hidden_state.shape[0]), context_offsets, :]
        example_tensor = example_outputs.last_hidden_state[torch.arange(0, example_outputs.last_hidden_state.shape[0]), example_offsets,:]
        sim = self.cosine(context_tensor, example_tensor)
        return sim
    
class CoreModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = SimilarityModule()
        self.scorer = ScoreModule([256])
    def forward(self, data):
        for param in self.sim.parameters():
            param.requires_grad = False
        x = self.sim(data)
        x = torch.unsqueeze(x,1)
        x = self.scorer(x)
        return x.to(device, dtype=torch.float32)
        
class WordSenseData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"average": self.data.loc[idx, "average"],
                "index": self.data.loc[idx, 'index'],
                "homonym": self.data.loc[idx, 'homonym'],
                "context": self.data.loc[idx, 'context'],
                "example_sentence": self.data.loc[idx, 'example_sentence']}
        

def train(model, train_set: Dataset, dev_set: Dataset, n_epochs: int = 100, batch_size = 64):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    train_loader = DataLoader(train_set, 
                        batch_size=batch_size, 
                        shuffle=True,)
    dev_loader = DataLoader(dev_set, 
                        batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    best_vloss = 1_000_000.
    for epoch in tqdm(range(n_epochs)):
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            y_batch = batch.pop("average")
            y_batch = torch.Tensor(y_batch)-1
            X_batch = batch
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch.to(device,
                                            dtype=torch.long))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss/len(train_loader)
        running_vloss = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        acc = 0
        with torch.no_grad():
            for vdata in dev_loader:
                vlabels = vdata.pop("average")
                vlabels = torch.Tensor(vlabels)-1
                vinputs = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels.to(device,
                                                    dtype=torch.long))
                acc += sum([a==b for a, b in zip(torch.argmax(voutputs, dim = 1), vlabels.to(dtype=torch.long))])
                running_vloss += vloss

        avg_vloss = running_vloss/len(dev_loader)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACCURACY valid {}'.format(acc/len(dev_set)))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/ambirt_{}_{}.safetensors'.format(timestamp,
                                                                epoch)
            save_model(model, model_path)
            
    return model_path

def run(model, data: pd.DataFrame):
    loader = DataLoader(data, 
                    batch_size= 64,)
    res = pd.DataFrame(columns = ["id", "prediction"])
    with torch.no_grad():
        for batch in loader:
            pred = torch.argmax(model(batch), dim=1)
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
        print("No data file was provided.")
        exit(1)
    else:
        train_set = load_data(argv[1])
        dev_set = load_data(argv[2])
    train_set = WordSenseData(train_set)
    dev_set = WordSenseData(dev_set)
    
    ## Model Running
    model = CoreModule()
    # model = SimilarityModule()
    # model_path = train(model, train_set, dev_set)
    model_path = "/Users/local/Documents/GitHub/SemEval52026/models/ambirt_20260103_005628_0.safetensors"
    load_model(model, model_path)
    res = run(model, dev_set)
    
    ## Saving Results
    res.to_json("predictions.jsonl",orient="records",lines=True)