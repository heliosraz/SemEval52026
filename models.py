from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer 
from load_data import load_data
import torch
from bisect import bisect
from typing import List, Dict
import sys


if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

class ScoreModule(torch.nn.Module):

    def __init__(self, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            layers = [torch.nn.Linear(1, hidden_sizes[0])]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(torch.nn.Linear(h_i, h_j))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_sizes[-1], 5))
            self.layers = torch.nn.Sequential(*layers)
        else:
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 5))
            
    def forward(self, x):
        return self.layers(x)

class ContentOffsetModule(torch.nn.Module):
    def __init__(self, model_name = "google-bert/bert-base-cased"):
        super().__init__()
        self.bert_layer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_offsets(self,data:Dict, feature_name, tok_outs):
        res = []
        offsets = tok_outs.pop('offset_mapping')
        # for each data instance find the offset
        for target, instance, instance_offsets in zip(data['homonym'], data[feature_name], offsets):
            offset = instance_offsets.tolist()
            word_loc = instance.find(target)
            word_off_ind = bisect(offset,
                                [word_loc, word_loc],
                                lo = 0,
                                hi = offset.index([0,0], 1))
            i, j = instance_offsets[word_off_ind]
            if i<=word_loc<=j:
                res.append(word_off_ind)
            else:
                res.append(word_off_ind-1)
        return torch.Tensor(res).long()

    def find_offset(self,target:str, text:str, offsets:list):
        # locate start and end of word in text
        word_loc = text.find(target)
        word_offset = [word_loc, word_loc + len(target)]
        # do binary search to find token offsets containing end of word
        return self.binary_search(offsets, word_offset[1])

    def binary_search(self,offsets, value):
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
        raise ValueError("Homonym was not found in instance.")

    def forward(self, data: Dict, select = ['context', 'example_sentence']):
        # taken from benchmark_bert.py
        toks = {}
        offsets = {}
        outputs = {}
        res_tensors = {}
        for param in select:
            toks[param] = self.tokenizer(list(data[param]),
                                    return_tensors='pt', 
                                    padding=True,
                                    return_offsets_mapping=True).to(device)
            # get list of target offsets for each data instance for both context and example
            offsets[param] = self.get_offsets(data,param, toks[param])
            outputs[param] = self.bert_layer(**toks[param])
            res_tensors[param] = outputs[param].last_hidden_state[torch.arange(0, outputs[param].last_hidden_state.shape[0]), offsets[param], :]
        return res_tensors

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


class SimilarityScoreModule(torch.nn.Module):

    def __init__(self,
                model_name = "google-bert/bert-base-cased",
                use_sbert:bool = False):
        self.use_sbert = use_sbert
        super().__init__()
        if use_sbert:
            self.sim = Sentence_SimModule(device, model_name='all-MiniLM-l6-v2')
        else:
            self.offset = ContentOffsetModule(model_name)
            self.sim = torch.nn.CosineSimilarity()
            for param in self.offset.parameters():
                param.requires_grad = False
        for param in self.sim.parameters():
            param.requires_grad = False
        self.scorer = ScoreModule([128])

    def forward(self, data, select = ["context", "example_sentence"]):
        if self.use_sbert:
            x = self.sim(data)
        else:
            x = self.offset(data, select)
            x = self.sim(x[select[0]], x[select[1]])
        x = x.unsqueeze(1).unsqueeze(1)
        y = self.scorer(x)
        return y.transpose(1, 2).squeeze(-1)
    
class CrossContextModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-cased")
        self.bert_tokenizer = AutoModel.from_pretrained("google-bert/bert-base-cased")
    def forward(self, data:Dict):
        context_toks = self.tokenizer(list(data['context']),
                                    return_tensors='pt', 
                                    padding=True).to(device)
        context_embeds = self.bert()
        
        

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






