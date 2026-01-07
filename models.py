from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer 
from load_data import load_data
import torch
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



class ScoreModule(torch.nn.Module):

    def __init__(self, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            layers = [torch.nn.Linear(1, hidden_sizes[0])]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(torch.nn.Linear(h_i, h_j))
                torch.nn.init.xavier_uniform_(self.layers[-1].weight, gain=5.0)
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.BatchNorm1d(h_j))
            layers.append(torch.nn.Linear(hidden_sizes[-1], 5))
            self.layers = torch.nn.Sequential(*layers)
        else:
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 5))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight, gain=5.0)
        

    def forward(self, x):
        return self.layers(x)

class SimilarityModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_layer = AutoModel.from_pretrained("google-bert/bert-base-cased")
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
        context_tensor = context_outputs.last_hidden_state[torch.arange(0, context_outputs.last_hidden_state.shape[0]), context_offsets, :]
        example_tensor = example_outputs.last_hidden_state[torch.arange(0, example_outputs.last_hidden_state.shape[0]), example_offsets,:]
        sim = self.cosine(context_tensor, example_tensor)
        return sim

class Sentence_SimModule(torch.nn.Module):
    # similarity module using SentenceTransformers for entire sentence embedding
    # instead of just target word
    
    def __init__(self, device, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)

    def forward(self,data:Dict):
        context_embeds = self.sbert_model.encode(list(data['context']),
                                                convert_to_tensor=True).to(self.device)
        example_embeds = self.sbert_model.encode(list(data['example_sentence']),
                                                convert_to_tensor=True).to(self.device)
        # 'similarity_pairwise' bc otherwise it's every context with every example
        sim = self.sbert_model.similarity_pairwise(context_embeds, example_embeds).to(self.device)
        return sim


class CoreModule(torch.nn.Module):

    def __init__(self, use_sbert:bool = False):
        super().__init__()
        if use_sbert:
            self.sim = Sentence_SimModule(device, model_name='all-MiniLM-l6-v2')
        else:
            self.sim = SimilarityModule()
        self.scorer = ScoreModule()

    def forward(self, data):
        for param in self.sim.parameters():
            param.requires_grad = False
        x = self.sim(data)
        x = torch.unsqueeze(x,1)
        x = self.scorer(x)
        return x

## these are for using sbert without explicit similarity metric (i.e feeding embeddings directly to ffn)
class NoSimSentenceModule(torch.nn.Module):
    # sentence embedding module without similarity 
    def __init__(self, device, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)

    def forward(self,data:Dict):
        context_embeds = self.sbert_model.encode(list(data['context']), convert_to_tensor=True).to(self.device)
        example_embeds = self.sbert_model.encode(list(data['example_sentence']), convert_to_tensor=True).to(self.device)
        return context_embeds, example_embeds

class NoSimScoreModule(torch.nn.Module):
    # embed size tbd based on combining sbert embeds
    # TODO: FIGURE OUT EMBED SIZE
    def __init__(self, device, embed_size, hidden_size):
        super().__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(embed_size, hidden_size).to(self.device)
        self.activation = torch.nn.ReLU().to(self.device)
        self.linear2 = torch.nn.Linear(hidden_size, 5).to(self.device)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class NoSimCoreModule(torch.nn.Module):
# TODO: figure out embed size by deciding on how to combine sbert embeddings
    # for now it's just 
    
    def __init__(self, device, hidden_size=256):
        self.device = device
        self.sentence_embed = NoSimSentenceModule(device=self.device, model_name ='all-MiniLM-L6-v2')
        self.scorer = NoSimScoreModule






