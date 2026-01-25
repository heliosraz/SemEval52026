from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer 
from data_processing import load_data
import torch
from bisect import bisect
from typing import List, Dict
import sys
import math
from random import randint


if torch.cuda.is_available():
    device = torch.device("cuda") 
elif torch.mps.is_available():
    device = torch.device("mps")
else: 
    device = torch.device("cpu")

########## Submodules

class ClassifierModule(torch.nn.Module):
    """Scores input signal on a scale of 5 

    Args:
        input_len (int): input features
        hidden_sizes (List[int]): list of hidden layer sizes
    """
    def __init__(self, input_len = 1, output_len = 5, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            layers = [torch.nn.Linear(input_len, hidden_sizes[0])]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(torch.nn.Linear(h_i, h_j))
            layers.append(torch.nn.Linear(hidden_sizes[-1], output_len))
            self.layers = torch.nn.ModuleList(layers)
        else:
            self.layers = torch.nn.ModuleList([torch.nn.Linear(input_len, output_len)])
        self.train_mode = False
    
    def train(self):
        self.train_mode = True
    
    def eval(self):
        self.train_mode = False
            
    def forward(self, x, drop_p=0.3):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)
            x = torch.dropout(x, drop_p, train = self.train_mode)
        return self.layers[-1](x)

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
    """Generates embedding (and offsets) using a given Bert-like model.
    Handles all embedding related tasks.

    Args:
        model_name (str): name of a Bert-like model to use (e.g. "google-bert/bert-base-cased")
        max_length (int): max full_context length of the model used in padding or truncation.
    """
    def __init__(self,
                model_name = "google-bert/bert-base-cased",
                max_length = 512):
        super().__init__()
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_embedding_size(self):
        return self.model.config.hidden_size
        
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
    
    def mask(self, data, tokens):
        # find how long each sequence is
        instances = self.tokenizer(data,
                    padding = False).to(device)
        # print([len(toks) for toks in instances], max([len(toks) for toks in instances]))
        # generate a random masked index (-1 being no mask)
        mask_res = {"mask_ids": [], "mask_inds": [randint(-1, len(toks)-1) for toks in instances["input_ids"]]}
        # for each masking index generate a pseudo-mask
        # or gather the true token id and mask
        for i, replace_ind in enumerate(mask_res["mask_inds"]):
            if replace_ind<0:
                replace_ind = randint(0, len(instances["input_ids"][i])-1)
                mask_res["mask_inds"][i] = replace_ind
                true_token = instances["input_ids"][i][replace_ind]
            else:
                true_token = instances["input_ids"][i][replace_ind]
                tokens["input_ids"][i][replace_ind] = self.tokenizer.mask_token_id
            mask_res["mask_ids"].append(true_token)
        return tokens, mask_res
    
    def forward(self, data, offset = False, tokenize = True):
        if tokenize:
            tokens = self.tokenizer(data,
                                    return_tensors='pt',
                                    padding = "max_length",
                                    max_length = self.max_length,
                                    return_offsets_mapping=offset).to(device)
        else:
            tokens = data
        if offset:
            offsets = tokens.pop("offset_mapping")
                
        embeds = self.model(**tokens)
        embeds = embeds.last_hidden_state
        if offset:
            return embeds, offsets
        else:
            return embeds
            
class ContextOffsetModule(torch.nn.Module):
    """Redundant consolidation of ContextEmbedModule
    TODO: refactor s.t. all instances of this module uses ContextEmbedModule
    """
    # separate from BERT module to test this with different models (e.g. LLMs)
    def __init__(self, model_name = "google-bert/bert-base-cased"):
        super().__init__()
        self.embed = ContextEmbedModule(model_name)

    def forward(self, data: Dict, select = ['full_context', 'example_sentence']):
        # taken from benchmark_bert.py
        offsets = {}
        outputs = {}
        res_tensors = {}
        for param in select:
            outputs[param], batch_offsets = self.embed(list(data[param]), include_offsets = True)
            # get list of target offsets for each data instance for both full_context and example
            offsets[param] = self.embed.get_offsets(data, param, batch_offsets)
            res_tensors[param] = outputs[param][torch.arange(0, outputs[param].shape[0]), offsets[param], :]
        return res_tensors

class SentenceEmbedModule(torch.nn.Module):
    """Embeds a sentence using SentenceTransformers

    Args:
        model_name (str): name of SBert-like model (e.g. sentencetransformers/all-MiniLM-L6-v2')
    """
    def __init__(self, model_name='sentencetransformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.sbert_model = SentenceTransformer(model_name, device=device)
    
    def forward(self, data):
        embeds = self.sbert_model.encode(data,
                                        convert_to_tensor=True).to(device)
        return embeds

########### Main Modules
class Sentence_SimModule(torch.nn.Module):
    """Similarity module using SentenceTransformers for entire sentence embedding instead of just target word

    Args:
        device (str): 'cpu', 'mps', or 'cuda'
        model_name (str): Sbert-like model name 
    """
    
    def __init__(self, device, model_name='sentencetransformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)

    def forward(self,data:Dict):
        context_embeds = self.sbert_model.encode(list(data['full_context']),
                                                convert_to_tensor=True).to(self.device)
        example_embeds = self.sbert_model.encode(list(data['example_sentence']),
                                                convert_to_tensor=True).to(self.device)
        # 'similarity_pairwise' bc otherwise it's every full_context with every example
        sim = self.sbert_model.similarity_pairwise(context_embeds, example_embeds).to(self.device)
        return sim

class SimilarityScoreModule(torch.nn.Module):
    """Connects similarity module with a score component
    Args:
        model_name (str): Bert/Sbert-like model name 
        use_sbert (bool): whether to use Sentence_SimModule for embeddings
    """
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
            self.offset = ContextOffsetModule(model_name)
            self.sim = torch.nn.CosineSimilarity()
            for param in self.offset.parameters():
                param.requires_grad = False
        self.scorer = ClassifierModule(hidden_sizes = [128])

    def forward(self, data, select = ["full_context", "example_sentence"]):
        if self.use_sbert:
            x = self.sim(data)
        else:
            x = self.offset(data, select)
            x = self.sim(x[select[0]], x[select[1]])
        x = x.unsqueeze(1).unsqueeze(1)
        y = self.scorer(x)
        return y.transpose(1, 2).squeeze(-1)
    
class CrossContentSimilarityModule(torch.nn.Module):
    """Compares a sentence embedding of the candidate sentence with all contextual embeddings of the full_context excerpt.

    Args:
        model_name (str): Sbert-like model name
        max_length (int): max full_context length for the model
    """
    def __init__(self,
                model_name = "sentence-transformers/all-MiniLM-L6-v2",
                max_length = 512):
        super().__init__()
        self.max_length = max_length
        self.context_former = ContextEmbedModule(model_name = model_name, max_length = max_length)
        self.sentence_former = SentenceEmbedModule(model_name = model_name)
        
        for model in [self.context_former, self.sentence_former]:
            for param in model.parameters():
                param.requires_grad = False
                
        self.sim = torch.nn.CosineSimilarity(dim = 2)
        self.scorer = ClassifierModule(input_len = max_length, hidden_sizes = [128])
    def forward(self, data:Dict, select = ["full_context", "judged_meaning"]):
        # full_context gets fed into bert to get contextual embeddings
        content_embed = self.context_former(data[select[0]])
        # example_sentence/definition fed into sbert to get pooled contextual embeddings
        candidate_embed = self.sentence_former(data[select[1]]).unsqueeze(-1)
        # get similarity of candidate with each of content_embed
        # similarities = self.sim(content_embed, candidate_embed)
        similarities = torch.bmm(content_embed, candidate_embed).transpose(1,2)
        # feed similarities into scorer
        y = self.scorer(similarities)
        return y
    
class GeneralistModel_nosep(torch.nn.Module):
    """GLiNER inspired designed module

    Args:
        model_name (str): Bert-like model name
        max_length (int): max full_context length for the model
    """
    def __init__(self,
            model_name = "google-bert/bert-base-cased",
            max_length = 512,
            d_attn = 128,
            dropout_p = 0.4):
        super().__init__()
        self.name = model_name
        self.model = ContextEmbedModule(model_name = model_name,
                                        max_length = max_length)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        self.max_length = max_length
        n = self.model.get_embedding_size()
        self.K = torch.nn.Linear(n, d_attn, bias=False)
        self.Q = torch.nn.Linear(n, d_attn, bias=False)
        self.V = torch.nn.Linear(n, d_attn, bias=False)
        torch.nn.init.xavier_normal_(self.K.weight)  
        torch.nn.init.xavier_normal_(self.Q.weight) 
        torch.nn.init.xavier_normal_(self.V.weight) 
        self.drop_attn = dropout_p
        self.train_mode = False
    
    def train(self):
        self.train_mode = True
    
    def eval(self):
        self.train_mode = False
        
    def get_vocab_size(self):
        return self.model.tokenizer.vocab_size
        
    def scaled_dot_product_attention(
                        self,
                        query,
                        key,
                        value,
                        attn_mask=None,
                        dropout_p=0.3,
                        is_causal=False,
                        scale=None,
                        enable_gqa=False) -> torch.Tensor:
        b, L, S = query.size(0), query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(b, L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-1e9"))
            else:
                attn_mask = attn_mask.bool()
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-1e9"))

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim = -1) 
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.train_mode)
        return attn_weight @ value
        
    def forward(self, data:Dict, select = ["full_context", "judged_meaning"], mask = False):
        candidate_toks = self.model.tokenizer(
                                data[select[1]],
                                return_tensors = "pt",
                                padding = "max_length",
                                max_length = self.max_length//2).to(device)
        if mask:
            candidate_toks, masks = self.model.mask(data[select[1]], candidate_toks)
        context_toks = self.model.tokenizer(
                                data[select[0]],
                                return_tensors = "pt",
                                padding = "max_length",
                                max_length = self.max_length//2).to(device)
        # separate by [SEP] and feed each into refiner
        candidate_embeds = self.model(
                                candidate_toks,
                                tokenize = False)
        context_embeds = self.model(
                                context_toks,
                                tokenize = False)
        # feed into refiner
        refined_context = self.K(context_embeds)
        refined_candidate = self.Q(candidate_embeds)
        aggre_value = self.V(context_embeds)
        attn_mask = torch.bmm(
                        candidate_toks["attention_mask"].float().unsqueeze(2),
                        context_toks["attention_mask"].float().unsqueeze(1)
                        )
        
        # scaled dot product with sigmoid
        x = self.scaled_dot_product_attention(
            query = refined_candidate,
            key = refined_context,
            value = aggre_value,
            attn_mask = attn_mask,
            dropout_p=self.drop_attn
            )
        
        return x, masks
    
    
class GeneralistModel(torch.nn.Module):
    """GLiNER inspired designed module

    Args:
        model_name (str): Bert-like model name
        max_length (int): max full_context length for the model
    """
    def __init__(self,
            model_name = "google-bert/bert-base-cased",
            max_length = 512,
            d_attn = 128,
            dropout_p = 0.4):
        super().__init__()
        self.name = model_name
        self.model = ContextEmbedModule(model_name = model_name,
                                        max_length = max_length)
        for param in self.model.parameters():
            param.requires_grad = False
        self.max_length = max_length
        n = self.model.get_embedding_size()
        self.K = torch.nn.Linear(n, d_attn, bias=False)
        self.Q = torch.nn.Linear(n, d_attn, bias=False)
        self.V = torch.nn.Linear(n, d_attn, bias=False)
        torch.nn.init.xavier_normal_(self.K.weight)  
        torch.nn.init.xavier_normal_(self.Q.weight) 
        torch.nn.init.xavier_normal_(self.V.weight)
        self.train_mode = False
        self.drop_attn = dropout_p
    
    def train(self):
        self.train_mode = True
    
    def eval(self):
        self.train_mode = False
        
    def get_vocab_size(self):
        return self.model.tokenizer.vocab_size
        
    def scaled_dot_product_attention(
                        self,
                        query,
                        key,
                        value,
                        attn_mask=None,
                        dropout_p=0.3,
                        is_causal=False,
                        scale=None,
                        enable_gqa=False) -> torch.Tensor:
        b, L, S = query.size(0), query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(b, L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-1e9"))
            else:
                attn_mask = attn_mask.bool()
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-1e9"))

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim = -1) 
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.train_mode)
        return attn_weight @ value
        
    def forward(self, data:Dict, select = ["full_context", "judged_meaning"], mask = False):
        data["sep"] = [self.model.tokenizer.sep_token]*len(data[select[0]])
        sep_toks = self.model.tokenizer(
                                data["sep"],
                                return_tensors = "pt",
                                padding = False).to(device)
        sep_size = len(sep_toks["input_ids"][0])
        candidate_toks = self.model.tokenizer(
                                data[select[1]],
                                return_tensors = "pt",
                                padding = "max_length",
                                max_length = self.max_length//2-sep_size,).to(device)
        if mask:
            candidate_toks, masks = self.model.mask(data[select[1]], candidate_toks)
        context_toks = self.model.tokenizer(
                                data[select[0]],
                                return_tensors = "pt",
                                padding = "max_length",
                                max_length = self.max_length//2).to(device)
        sep_toks["attention_mask"] = torch.zeros(
                                        sep_toks["attention_mask"].shape).to(device)
        input_seq = {}
        input_seq["input_ids"] = torch.concat(
                                        [candidate_toks["input_ids"],
                                        sep_toks["input_ids"],
                                        context_toks["input_ids"]],
                                        axis = 1)
        input_seq["attention_mask"] = torch.concat(
                                        [candidate_toks["attention_mask"],
                                        sep_toks["attention_mask"],
                                        context_toks["attention_mask"]],
                                        axis = 1).bool()
        # separate by [SEP] and feed each into refiner
        input_embeds = self.model(
                                input_seq,
                                tokenize = False)
        sep_inds = [(len(candi), len(candi)+len(sep)) 
                    for candi, sep in zip(
                        candidate_toks["input_ids"],
                        sep_toks["input_ids"])]
        
        context_embeds = []
        candidate_embeds = []
        for i in range(input_embeds.shape[0]):
            start, end = sep_inds[i]
            candidate_embeds.append(input_embeds[i, :start, :])
            context_embeds.append(input_embeds[i, end:, :])
        context_embeds = torch.stack(context_embeds)
        candidate_embeds = torch.stack(candidate_embeds)
        
        # feed into refiner
        refined_context = self.K(context_embeds)
        refined_candidate = self.Q(candidate_embeds)
        aggre_value = self.V(context_embeds)
        attn_mask = torch.bmm(
                        candidate_toks["attention_mask"].float().unsqueeze(2),
                        context_toks["attention_mask"].float().unsqueeze(1)
                        )
        
        # scaled dot product with sigmoid
        x = self.scaled_dot_product_attention(
            query = refined_candidate,
            key = refined_context,
            value = aggre_value,
            attn_mask = attn_mask,
            dropout_p=self.drop_attn
            )
        
        return x, masks
    
class GeneralistModelScored(torch.nn.Module):
    def __init__(self,
            model_name = "google-bert/bert-base-cased",
            max_length = 512,
            d_attn = 128,
            drop_attn = 0.4,
            drop_cls = 0.4):
        super().__init__()
        self.name = model_name
        self.base_model = GeneralistModel(model_name, max_length, d_attn, dropout_p=drop_attn)
        self.classifier = ClassifierModule(input_len = d_attn, hidden_sizes = [])
        self.train_mode = False
        self.drop_cls = drop_cls
    
    def train(self):
        self.base_model.train()
        self.train_mode = True
    
    def eval(self):
        self.base_model.eval()
        self.train_mode = False
        
    def forward(self, data, select = ["full_context", "judged_meaning"], mask = False):
        x, mask_res = self.base_model(data, select, mask=mask)
        x = x.max(dim = 1)[0]
        y = self.classifier(x, self.drop_cls)
        return y, mask_res
    
class PretrainedGeneralistModel(torch.nn.Module):
    def __init__(
        self,
        base = GeneralistModel,
        model_name = "google-bert/bert-base-cased",
        max_length = 512,
        d_attn = 128,
        drop_attn = 0.4,
        drop_cls = 0.4):
        super().__init__()
        self.name = model_name
        self.base_model = base(model_name, max_length, d_attn, dropout_p = drop_attn)
        vocab_size = self.base_model.get_vocab_size()
        self.classifier = ClassifierModule(
            input_len = d_attn,
            output_len = vocab_size)
        self.train_mode = False
        self.drop_cls = drop_cls
    
    def train(self):
        self.base_model.train()
        self.train_mode = True
    
    def eval(self):
        self.base_model.eval()
        self.train_mode = False
        
    def forward(self, data, select = ["full_context", "judged_meaning"], mask = True):
        x, mask_res = self.base_model(data, mask = mask, select = select) # [batch, q_seq, d_embed] x [batch, k_seq, d_embed].T x [batch, k_seq, d_embed] -> [16, q_seq, d_attn]
        if mask:
            mask_inds = mask_res["mask_inds"]
            x = x[torch.arange(x.shape[0]), mask_inds, :] # [batch, q_seq, d_attn] -> [16, 1, d_attn]
        else:
            x = x.max(dim = 1)[0].unsqueeze(1)
        y = self.classifier(x, self.drop_cls) # [16, 1, d_attn] -> [16, class_out]
        return y, mask_res

class SynonymModule(GeneralistModel):
    """TODO docstring"""
    
    def __init__(self,
                 model_name: str = "google-bert/bert-base-cased",
                 max_length: int = 512,
                 d_attn: int = 128,
                 dropout_p: float = 0.4,
                 drop_cls: float = 0.4):
        super().__init(model_name, max_length, d_attn, dropout_p)
        self.name = model_name
        self.model = ContextEmbedModule(model_name, max_length)
        self.fc = ClassifierModule(input_len=d_attn, hidden_sizes=[])
        self.drop_cls = drop_cls
        self.max_length = max_length
        n = self.model.get_embedding_size()
        
        self.K = torch.nn.Linear(n, d_attn, bias=False)
        self.Q = torch.nn.Linear(n, d_attn, bias=False)
        self.V = torch.nn.Linear(n, d_attn, bias=False)
        torch.nn.init.xavier_normal_(self.K.weight)
        torch.nn.init.xavier_normal_(self.Q.weight)
        torch.nn.init.xavier_normal_(self.V.weight)
        self.train_mode = False
        self.drop_attn = dropout_p

        

    def forward(self, data: Dict, select=["full_context", "homonym"], mask=False):
        data["sep"] = [self.model.tokenizer.sep_token]*len(data)

        sep_toks = self.model.tokenizer(data["sep"], return_tensors = "pt", padding=False).to(device)
        sep_size = len(sep_toks["input_ids"][0])
        
        synonyms = list(map(self.wordnet_synonyms, data[select[1]]))
        syn_toks = self.model.tokenizer(synonyms, return_tensors="pt",padding=False).to(device)

        context_toks = self.model.tokenizer(data[select[0]],
                                            return_tensors="pt",padding=False).to(device)
        word_toks = self.model.tokenizer(data[select[1]],
                                         return_tensors="pt", padding=False).to(device) 

        input_seq = {"input_ids": torch.concat([syn_toks["input_ids"],
                                                sep_toks["input_ids"],
                                                context_toks["input_ids"],
                                                word_toks["input_ids"]]),
                     "attention_mask": torch.concat([syn_toks["attention_mask"],
                                                    sep_toks["attention_mask"],
                                                    context_toks["attention_mask"],
                                                    word_toks["attention_mask"]])}

                
        
        #TODO post-refiner attn layers
        refined_ctx = self.K(context_embeds)
        refined_syns = self.Q(cand_embeds)
        aggre_value = self.V(context_embeds)
        #TODO attn mask


        x = self.scaled_dot_product_attention(refined_syns,
                                              refined_ctx,
                                              aggre_value,
                                              attn_mask=attn_mask,
                                              dropout_p=self.drop_attn)
        return x, masks
    def wordnet_synonyms(self, w: str) -> List[str]:
        """TODO docstring"""
        syns = wn.synonyms(w)
        pairs = list(zip(["[SNS] " ] * len(syns), set(f"{wd[0]}" for wd in syns if wd)))
        pairs.append(("[SNS] ", "[MISC]"))
        return "".join("".join(p) for p in pairs)
    
## these are for using sbert without explicit similarity metric (i.e feeding embeddings directly to ffn)
class NoSimSentenceModule(torch.nn.Module):
    # sentence embedding module without similarity 
    def __init__(self, device, model_name='all-MiniLM-L6-v2'):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)

    def forward(self,data:Dict):
        context_embeds = self.sbert_model.encode(list(data['full_context']), convert_to_tensor=True).to(self.device)
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






