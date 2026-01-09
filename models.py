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

########## Submodules

class ScoreModule(torch.nn.Module):

    def __init__(self, input_len = 1, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            layers = [torch.nn.Linear(input_len, hidden_sizes[0])]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(torch.nn.Linear(h_i, h_j))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_sizes[-1], 5))
            self.layers = torch.nn.Sequential(*layers)
        else:
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 5))
            
    def forward(self, x):
        return self.layers(x)

class RefineModule(torch.nn.Module):
    def __init__(self, input_len = 1, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            layers = [torch.nn.Linear(input_len, hidden_sizes[0])]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(torch.nn.Linear(h_i, h_j))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_sizes[-1], 5))
            self.layers = torch.nn.Sequential(*layers)
        else:
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 5))
            
    def forward(self, x):
        return self.layers(x)
    
class ContextEmbedModule(torch.nn.Module):
    def __init__(self,
                model_name = "google-bert/bert-base-cased",
                max_length = 512):
        super().__init__()
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_offsets(self, data:Dict, feature_name, offsets, tar_name = 'homonym'):
        res = []
        # for each data instance find the offset
        for target, instance, instance_offsets in zip(data[tar_name], data[feature_name], offsets):
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
    
    def forward(self, data, include_offsets = False):
        tokens = self.tokenizer(data,
                                return_tensors='pt',
                                padding = "max_length",
                                max_length = self.max_length,
                                return_offsets_mapping=include_offsets).to(device)
        if include_offsets:
            offsets = tokens.pop("offset_mapping")
        else:
            offsets = torch.Tensor()
        embeds = self.model(**tokens)
        embeds = embeds.last_hidden_state
        if include_offsets:
            return embeds, offsets
        else:
            return embeds

class ContentOffsetModule(torch.nn.Module):
    # separate from BERT module to test this with different models (e.g. LLMs)
    def __init__(self, model_name = "google-bert/bert-base-cased"):
        super().__init__()
        self.embed = ContextEmbedModule(model_name)

    def forward(self, data: Dict, select = ['context', 'example_sentence']):
        # taken from benchmark_bert.py
        offsets = {}
        outputs = {}
        res_tensors = {}
        for param in select:
            outputs[param], batch_offsets = self.embed(list(data[param]), include_offsets = True)
            # get list of target offsets for each data instance for both context and example
            offsets[param] = self.embed.get_offsets(data, param, batch_offsets)
            res_tensors[param] = outputs[param][torch.arange(0, outputs[param].shape[0]), offsets[param], :]
        return res_tensors

class SentenceEmbedModule(torch.nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        self.sbert_model = SentenceTransformer(model_name, device=device)
    
    def forward(self, data):
        embeds = self.sbert_model.encode(data,
                                        convert_to_tensor=True).to(device)
        return embeds

########### Main Modules
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
            for param in self.sim.parameters():
                param.requires_grad = False
        else:
            self.offset = ContentOffsetModule(model_name)
            self.sim = torch.nn.CosineSimilarity()
            for param in self.offset.parameters():
                param.requires_grad = False
        self.scorer = ScoreModule(hidden_sizes = [128])

    def forward(self, data, select = ["context", "example_sentence"]):
        if self.use_sbert:
            x = self.sim(data)
        else:
            x = self.offset(data, select)
            x = self.sim(x[select[0]], x[select[1]])
        x = x.unsqueeze(1).unsqueeze(1)
        y = self.scorer(x)
        return y.transpose(1, 2).squeeze(-1)
    
class CrossContentSimilarityModule(torch.nn.Module):
    def __init__(self,
                cformer_name = "sentence-transformers/all-MiniLM-L6-v2",
                max_length = 512):
        super().__init__()
        self.max_length = max_length
        self.context_former = ContextEmbedModule(model_name = cformer_name, max_length = max_length)
        self.sentence_former = SentenceEmbedModule(model_name = cformer_name)
        
        for model in [self.context_former, self.sentence_former]:
            for param in model.parameters():
                param.requires_grad = False
                
        self.sim = torch.nn.CosineSimilarity(dim = 2)
        self.scorer = ScoreModule(input_len = max_length, hidden_sizes = [128])
    def forward(self, data:Dict, select = ["context", "judged_meaning"]):
        # context gets fed into bert to get contextual embeddings
        content_embed = self.context_former(data[select[0]])
        # example_sentence/definition fed into sbert to get pooled contextual embeddings
        candidate_embed = self.sentence_former(data[select[1]]).unsqueeze(1)
        # get similarity of candidate with each of content_embed
        similarities = self.sim(content_embed, candidate_embed)
        # feed similarities into scorer
        y = self.scorer(similarities)
        return y
        
class GeneralistModel(torch.nn.Module):
    def __init__(self,
            model_name = "google-bert/bert-base-cased"):
        super().__init__()
        self.model = ContextEmbedModule(model_name = model_name)
        self.scorer = ScoreModule(hidden_sizes = [128])
        self.content_refiner = RefineModule()
        self.candidate_refiner = RefineModule()
        
        
    def forward(self, data:Dict, select = ["context", "definition"]):
        input_seqs = []
        for context, candidate in zip(list(data[select[0]], list(data[select[1]]))):
            input_seqs.append(candidate+" [SEP] "+context)
        data["target"] = ["[SEP]"]*len(data[select[0]])
        data["input"] = input_seqs
        
        input_embeds, input_offsets= self.embed(input_seqs, include_offsets = True)
        # separate by [SEP] and feed each into refiner
        sep_inds = self.embed.get_offsets(data,
                                        "input",
                                        input_offsets,
                                        tar_name = 'target')
        context_embeds, candi_embeds = embed[:, :sep, :], embed[:, sep+1:, :]
        # dot product
        torch.einsum()
        #decode algo or scorer module
        return

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






