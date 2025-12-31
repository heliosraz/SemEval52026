from transformers import AutoTokenizer, AutoModel
from load_data import make_dataset
import torch
import tqdm
from sys import argv, exit
from torch.utils.data import DataLoader, Dataset
import pandas as pd

if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")
    
def tokenize(tokenizer, data):
    return tokenizer(list(data), return_tensors='pt', padding=True, return_offsets_mapping=True)

def get_all_offsets(data, **kwargs):
    res = {}
    # for each data instance find the target word's offset
    for name, context in kwargs.items():
        res[name] = []
        instance_offsets = context.pop('offset_mapping')
        for i, offsets in zip(range(len(data)), instance_offsets):
            instance = data.iloc[i]
            target = instance['homonym']
            res[name].append(find_offset(target, text=instance[name], offsets=offsets))
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
        self.linear1 = torch.nn.Linear(hidden_size, 1)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, hidden_size)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        x = torch.argmax(x)
        return x

class SimilarityModule(torch.nn.Module):
    # torch no grad should 

    def __init__(self):
        super().__init__()
        self.bert_layer = AutoModel.from_pretrained("google-bert/bert-base-cased").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased").to(device)
        self.cosine = torch.nn.CosineSimilarity(dim=0)

    def forward(self, data):
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
        context_offsets, example_offsets = get_all_offsets(context_toks, example_toks, data)
        context_outputs = self.bert_layer(**context_toks)
        example_outputs = self.bert_layer(**example_toks)
        sim = []
        for i in tqdm.tqdm(range(len(context_offsets))):
            context_tensor = context_outputs.last_hidden_state[i, context_offsets[i],:]
            example_tensor = example_outputs.last_hidden_state[i, example_offsets[i],:]
            sim.append(self.cosine(context_tensor,example_tensor) * 5)
        return sim
    
class CombinedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = SimilarityModule()
        self.scorer = ScoreModule(12)
    def forward(self, data):
        with torch.no_grad():
            x = self.sim(data)
        x = self.scorer(x)
        
class WordSenseData(Dataset):
    def __init__(self, data_dir):
        self.data = pd.read_json(data_dir)
    def __getitem__(self, idx):
        pass # TODO
        

def train(model, data: Dataset, epochs: int = 2):
    batches = DataLoader(data, 
                        batch_size= 64, 
                        shuffle=True,)
    for _ in range(epochs):
        for batch in batches:
            pass # TODO

def run(model, data: Dataset):
    batches = DataLoader(data, 
                    batch_size= 64, 
                    shuffle=True,)
    for batch in batches:
        pass # TODO
            
    

if __name__ == "__main__":
    ## Data Processing
    if len(argv)<1:
        print("No data file was provided")
        exit()
    else:
        data = WordSenseData(argv[1])
    model = CombinedModule()
    # train(data)
    sim = run(data)
    with open('results.txt', 'w') as f:
        f.writelines(sim)
