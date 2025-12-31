from transformers import AutoTokenizer, AutoModel
from load_data import load_data
import torch
import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset
import pandas as pd

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

    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, 5)
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        y = torch.argmax(x, dim = 1)
        return y

class SimilarityModule(torch.nn.Module):
    # torch no grad should 

    def __init__(self):
        super().__init__()
        self.bert_layer = AutoModel.from_pretrained("google-bert/bert-base-cased").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        self.cosine = torch.nn.CosineSimilarity(dim=0)

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
        for i in tqdm.tqdm(range(len(context_offsets))):
            # finding contextual embedding of target
            context_tensor = context_outputs.last_hidden_state[i, context_offsets[i],:]
            example_tensor = example_outputs.last_hidden_state[i, example_offsets[i],:]
            # calculate cosine similarity
            sim.append(self.cosine(context_tensor,example_tensor)*5)
        return sim
    
class CombinedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = SimilarityModule()
        self.scorer = ScoreModule(12)
    def forward(self, data):
        with torch.no_grad():
            x = self.sim(data)
        x = torch.Tensor(x)
        x = torch.unsqueeze(x,1)
        x = self.scorer(x)
        return x
        
class WordSenseData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"index": self.data.loc[idx, 'index'],
                "homonym": self.data.loc[idx, 'homonym'],
                "context": self.data.loc[idx, 'context'],
                "example_sentence": self.data.loc[idx, 'example_sentence']}
        

def train(model, data: List, n_epochs: int = 2):
    loader = DataLoader(data, 
                        batch_size=64, 
                        shuffle=True,)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def run(model, data: pd.DataFrame):
    loader = DataLoader(data, 
                    batch_size= 64,)
    res = pd.DataFrame(columns = ["id", "prediction"])
    with torch.no_grad():
        for batch in loader:
            y = pd.DataFrame(model(batch), columns=['prediction'])
            y['id'] = batch['index']

            res = pd.concat([res, y])
    res["id"] = res["id"].astype("str")
    res["prediction"] = res["prediction"].astype("int")
    return res
        
    

if __name__ == "__main__":
    ## Data Processing
    if len(argv)<1:
        print("No data file was provided.")
        exit(1)
    else:
        data = load_data(argv[1])
    data = WordSenseData(data)
        
    ## Model Running
    # model = CombinedModule()
    model = SimilarityModule()
    # train(data)
    sim = run(model, data)
    
    ## Saving Results
    sim.to_json("predictions.jsonl",orient="records",lines=True)
