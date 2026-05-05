from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from data_processing import load_data
import torch
from bisect import bisect
from typing import List, Dict
import sys
import math
import nltk

nltk.download("wordnet")
from nltk.corpus import wordnet as wn
from random import randint
from abc import ABC, abstractmethod


########## Submodules
class ClassifierModule(torch.nn.Module):
    """Scores input signal on a scale of 5

    Args:
        input_len (int): input features
        hidden_sizes (List[int]): list of hidden layer sizes
    """

    def __init__(self, input_len=1, output_len=5, dropout=0.3, hidden_sizes=[]):
        super().__init__()
        if hidden_sizes:
            layers = [torch.nn.Linear(input_len, hidden_sizes[0])]
            for h_i, h_j in zip(hidden_sizes, hidden_sizes[1:]):
                layers.append(torch.nn.Linear(h_i, h_j))
            layers.append(torch.nn.Linear(hidden_sizes[-1], output_len))
            self.layers = torch.nn.ModuleList(layers)
        else:
            self.layers = torch.nn.ModuleList([torch.nn.Linear(input_len, output_len)])
        self.dropout = dropout

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        return self.layers[-1](x)


class ContextEmbedModule(torch.nn.Module):
    """Generates embedding (and offsets) using a given Bert-like model.
    Handles all embedding related tasks.

    Args:
        model_name (str): name of a Bert-like model to use (e.g. "google-bert/bert-base-cased")
        max_length (int): max full_context length of the model used in padding or truncation.
    """

    def __init__(
        self, model_name="google-bert/bert-base-cased", max_length=512, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.model = AutoModel.from_pretrained(model_name, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, device_map=self.device
        )

    def get_embedding_size(self):
        return self.model.config.hidden_size

    def get_offsets(self, data: Dict, feature_name, offsets, tar_name="homonym"):
        res = []
        # for each data instance find the offset
        for target, instance, instance_offsets in zip(
            data[tar_name], data[feature_name], offsets
        ):
            offset = instance_offsets.tolist()
            word_loc = instance.find(target)
            word_off_ind = bisect(
                offset, [word_loc, word_loc], lo=0, hi=offset.index([0, 0], 1)
            )
            i, j = instance_offsets[word_off_ind]
            if i <= word_loc <= j:
                res.append(word_off_ind)
            else:
                res.append(word_off_ind - 1)
        return torch.Tensor(res).long()

    def mask(self, data, tokens):
        # find how long each sequence is
        instances = self.tokenizer(data, padding=False).to(self.device)
        # print([len(toks) for toks in instances], max([len(toks) for toks in instances]))
        # generate a random masked index (-1 being no mask)
        mask_res = {
            "mask_ids": [],
            "mask_inds": [
                randint(-1, len(toks) - 1) for toks in instances["input_ids"]
            ],
        }
        # for each masking index generate a pseudo-mask
        # or gather the true token id and mask
        for i, replace_ind in enumerate(mask_res["mask_inds"]):
            if replace_ind < 0 or replace_ind >= len(tokens["input_ids"][i]):
                replace_ind = randint(0, len(instances["input_ids"][i]) - 1)
                mask_res["mask_inds"][i] = replace_ind
                true_token = instances["input_ids"][i][replace_ind]
            else:
                true_token = instances["input_ids"][i][replace_ind]
                tokens["input_ids"][i][replace_ind] = self.tokenizer.mask_token_id
            mask_res["mask_ids"].append(true_token)
        return tokens, mask_res

    def forward(self, data, offset=False, tokenize=True):
        if tokenize:
            tokens = self.tokenizer(
                data,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                return_offsets_mapping=offset,
            ).to(self.device)
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
    def __init__(self, model_name="google-bert/bert-base-cased", device="cpu"):
        super().__init__()
        self.device = device
        self.embed = ContextEmbedModule(model_name, device=self.device)

    def forward(self, data: Dict, select=["full_context", "example_sentence"]):
        # taken from benchmark_bert.py
        offsets = {}
        outputs = {}
        res_tensors = {}
        for param in select:
            outputs[param], batch_offsets = self.embed(list(data[param]), offset=True)
            # get list of target offsets for each data instance for both full_context and example
            offsets[param] = self.embed.get_offsets(data, param, batch_offsets)
            res_tensors[param] = outputs[param][
                torch.arange(0, outputs[param].shape[0]), offsets[param], :
            ]
        return res_tensors


class SentenceEmbedModule(torch.nn.Module):
    """Embeds a sentence using SentenceTransformers

    Args:
        model_name (str): name of SBert-like model (e.g. sentencetransformers/all-MiniLM-L6-v2')
    """

    def __init__(
        self,
        model_name="sentencetransformers/all-MiniLM-L6-v2",
        max_length=512,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)
        self.transformer = self.sbert_model[0]
        self.pooling = self.sbert_model[1]
        self.tokenizer = self.sbert_model.tokenizer
        self.max_length = max_length

    def forward(self, data):
        toks = self.tokenizer(
            data, padding="max_length", max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        # Get token embeddings
        output = self.transformer(toks)

        pooled_output = self.pooling(output)
        embeds = pooled_output["sentence_embedding"]  # Shape: (batch, hidden_dim)
        # embeds = self.sbert_model.encode(data, convert_to_tensor=True).to(device)
        return embeds


########### Main Modules


class BaselineModule(torch.nn.Module):
    """Baseline module that multiplies similarity scores by 5
    Args:
        model_name (str): Bert/Sbert-like model name
    """

    def __init__(
        self,
        model_name="google-bert/bert-base-cased",
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.model = ContextEmbedModule(model_name, device=self.device)
        self.sim = torch.nn.CosineSimilarity()

    def forward(self, data, select=["full_context", "example_sentence"]):
        offsets = {}
        outputs = {}
        res_tensors = {}
        for param in select:
            outputs[param], batch_offsets = self.model(list(data[param]), offset=True)
            # get list of target offsets for each data instance for both full_context and example
            offsets[param] = self.model.get_offsets(data, param, batch_offsets)
            res_tensors[param] = outputs[param][
                torch.arange(0, outputs[param].shape[0]), offsets[param], :
            ]
        x = self.sim(res_tensors[select[0]], res_tensors[select[1]])
        y = 5 * x
        return y.long()


class Sentence_SimModule(torch.nn.Module):
    """Similarity module using SentenceTransformers for entire sentence embedding instead of just target word

    Args:
        device (str): 'cpu', 'mps', or 'cuda'
        model_name (str): Sbert-like model name
    """

    def __init__(
        self, model_name="sentencetransformers/all-MiniLM-L6-v2", device="cpu"
    ):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)

    def forward(self, data: Dict, select=["full_context", "example_sentence"]):
        context_embeds = self.sbert_model.encode(
            list(data[select[0]]), convert_to_tensor=True
        ).to(self.device)
        example_embeds = self.sbert_model.encode(
            list(data[select[1]]), convert_to_tensor=True
        ).to(self.device)
        # 'similarity_pairwise' bc otherwise it's every full_context with every example
        sim = self.sbert_model.similarity_pairwise(context_embeds, example_embeds).to(
            self.device
        )
        return sim


class SimilarityScoreModule(torch.nn.Module):
    """Connects similarity module with a score component
    Args:
        model_name (str): Bert/Sbert-like model name
        use_sbert (bool): whether to use Sentence_SimModule for embeddings
    """

    def __init__(
        self,
        model_name="google-bert/bert-base-cased",
        use_sbert: bool = False,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.use_sbert = use_sbert
        if use_sbert:
            self.sim = Sentence_SimModule(
                model_name="all-MiniLM-l6-v2", device=self.device
            )
            for param in self.sim.parameters():
                param.requires_grad = False
        else:
            self.offset = ContextOffsetModule(model_name, device=self.device)
            self.sim = torch.nn.CosineSimilarity()
            for param in self.offset.parameters():
                param.requires_grad = False
        self.scorer = ClassifierModule(hidden_sizes=[128])

    def forward(self, data, select=["full_context", "example_sentence"]):
        if self.use_sbert:
            x = self.sim(data)
        else:
            x = self.offset(data, select)
            x = self.sim(x[select[0]], x[select[1]])
        x = x.unsqueeze(1).unsqueeze(1)
        y = self.scorer(x)
        return y.transpose(1, 2).squeeze(-1)


class CrossContextSimilarityModule(torch.nn.Module):
    """Compares a sentence embedding of the candidate sentence with all contextual embeddings of the full_context excerpt.

    Args:
        model_name (str): Sbert-like model name
        max_length (int): max full_context length for the model
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=512,
        drop_cls=0.3,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.context_former = ContextEmbedModule(
            model_name=model_name, max_length=max_length, device=self.device
        )
        self.tokenizer = self.context_former.tokenizer
        self.sentence_former = SentenceEmbedModule(
            model_name=model_name, device=self.device
        )

        self.sim = torch.nn.CosineSimilarity(dim=2)
        self.scorer = ClassifierModule(
            input_len=max_length, hidden_sizes=[128], dropout=drop_cls
        )

    def forward(
        self,
        data: Dict,
        select=["full_context", "judged_meaning"],
        return_sim=False,
        **kwargs,
    ):
        # full_context gets fed into bert to get contextual embeddings
        content_embed = self.context_former(data[select[0]])
        # example_sentence/definition fed into sbert to get pooled contextual embeddings
        candidate_embed = self.sentence_former(data[select[1]])
        # get similarity of candidate with each of content_embed
        # similarities = self.sim(content_embed, candidate_embed.unsqueeze(1)).unsqueeze(
        #     -1
        # )

        similarities = torch.bmm(content_embed, candidate_embed.unsqueeze(-1))
        if return_sim:
            return similarities.tolist(), similarities.shape
        # feed similarities into scorer
        y = self.scorer(similarities.transpose(1, 2))
        return y.squeeze(1)


class DXAModel_nosep(torch.nn.Module):
    """GLiNER inspired designed module

    Args:
        model_name (str): Bert-like model name
        max_length (int): max full_context length for the model
    """

    def __init__(
        self,
        model_name="google-bert/bert-base-cased",
        max_length=512,
        d_attn=768,
        dropout_p=0.4,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.name = model_name
        self.model = ContextEmbedModule(
            model_name=model_name, max_length=max_length, device=self.device
        )
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
        enable_gqa=False,
    ) -> torch.Tensor:
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
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.nn.functional.dropout(
            attn_weight, dropout_p, training=self.training
        )
        return attn_weight @ value

    def forward(
        self, data: Dict, select=["full_context", "judged_meaning"], mask=False
    ):
        candidate_toks = self.model.tokenizer(
            data[select[1]],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length // 2,
            add_special_tokens=False,
        ).to(self.device)
        if mask:
            candidate_toks, masks = self.model.mask(data[select[1]], candidate_toks)
        context_toks = self.model.tokenizer(
            data[select[0]],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length // 2,
            add_special_tokens=False,
        ).to(self.device)
        # separate by [SEP] and feed each into refiner
        candidate_embeds = self.model(candidate_toks, tokenize=False)
        context_embeds = self.model(context_toks, tokenize=False)
        # feed into refiner
        refined_context = self.K(context_embeds)
        refined_candidate = self.Q(candidate_embeds)
        aggre_value = self.V(context_embeds)
        attn_mask = torch.bmm(
            candidate_toks["attention_mask"].float().unsqueeze(2),
            context_toks["attention_mask"].float().unsqueeze(1),
        )

        # scaled dot product with sigmoid
        x = self.scaled_dot_product_attention(
            query=refined_candidate,
            key=refined_context,
            value=aggre_value,
            attn_mask=attn_mask,
            dropout_p=self.drop_attn,
        )

        return x, masks


class DXAModel(torch.nn.Module):
    """GLiNER inspired designed module

    Args:
        model_name (str): Bert-like model name
        max_length (int): max full_context length for the model
    """

    def __init__(
        self,
        model_name="google-bert/bert-base-cased",
        max_length=512,
        d_attn=768,
        dropout_p=0.4,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.name = model_name
        self.model = ContextEmbedModule(
            model_name=model_name, max_length=max_length, device=self.device
        )
        self.max_length = max_length
        n = self.model.get_embedding_size()
        self.K = torch.nn.Linear(n, d_attn, bias=False)
        self.Q = torch.nn.Linear(n, d_attn, bias=False)
        self.V = torch.nn.Linear(n, d_attn, bias=False)
        torch.nn.init.xavier_normal_(self.K.weight)
        torch.nn.init.xavier_normal_(self.Q.weight)
        torch.nn.init.xavier_normal_(self.V.weight)

        self.drop_attn = dropout_p

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
        enable_gqa=False,
    ) -> torch.Tensor:
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
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.nn.functional.dropout(
            attn_weight, dropout_p, training=self.training
        )
        return attn_weight @ value, attn_weight

    def forward(
        self,
        data: Dict,
        select=["full_context", "judged_meaning"],
        mask=False,
        return_sim=False,
    ):
        data["sep"] = [self.model.tokenizer.sep_token] * len(data[select[0]])
        sep_toks = self.model.tokenizer(
            data["sep"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).to(self.device)
        sep_size = len(sep_toks["input_ids"][0])
        candidate_toks = self.model.tokenizer(
            data[select[1]],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length // 2 - sep_size,
            add_special_tokens=False,
        ).to(self.device)
        if mask:
            candidate_toks, masks = self.model.mask(data[select[1]], candidate_toks)
        context_toks = self.model.tokenizer(
            data[select[0]],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length // 2,
            add_special_tokens=False,
        ).to(self.device)
        sep_toks["attention_mask"] = torch.zeros(sep_toks["attention_mask"].shape).to(
            self.device
        )
        input_seq = {}
        input_seq["input_ids"] = torch.concat(
            [
                candidate_toks["input_ids"],
                sep_toks["input_ids"],
                context_toks["input_ids"],
            ],
            axis=1,
        )
        input_seq["attention_mask"] = torch.concat(
            [
                candidate_toks["attention_mask"],
                sep_toks["attention_mask"],
                context_toks["attention_mask"],
            ],
            axis=1,
        ).bool()
        # separate by [SEP] and feed each into refiner
        input_embeds = self.model(input_seq, tokenize=False)
        sep_inds = [
            (len(candi), len(candi) + len(sep))
            for candi, sep in zip(candidate_toks["input_ids"], sep_toks["input_ids"])
        ]

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
            context_toks["attention_mask"].float().unsqueeze(1),
        )

        # scaled dot product with sigmoid
        if return_sim:
            x, sims = self.scaled_dot_product_attention(
                query=refined_candidate,
                key=refined_context,
                value=aggre_value,
                attn_mask=attn_mask,
                dropout_p=self.drop_attn,
            )
            return sims.flatten(-2).tolist(), sims.shape
        else:
            x, _ = self.scaled_dot_product_attention(
                query=refined_candidate,
                key=refined_context,
                value=aggre_value,
                attn_mask=attn_mask,
                dropout_p=self.drop_attn,
            )
        if mask:
            return x, masks
        return x


########### Wrapper Modules


class ModuleWrapper(torch.nn.Module, ABC):
    def __init__(
        self,
        base_type=DXAModel,
        base_name="google-bert/bert-base-cased",
        hidden_sizes=[50],
        max_length=512,
        d_attn=768,
        drop_attn=0.0,
        drop_cls=0.0,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.base_name = base_name
        self.base_model = base_type(
            base_name, max_length, d_attn, dropout_p=drop_attn, device=self.device
        )
        self.classifier = ClassifierModule(
            input_len=d_attn, hidden_sizes=hidden_sizes, dropout=drop_cls
        )
        self.tokenizer = self.base_model.model.tokenizer

    @abstractmethod
    def forward(self, data, select=["full_context", "judged_meaning"], mask=True):
        pass


class ScoredDXAModel(ModuleWrapper):
    def __init__(self, drop_cls=0.3, hidden_sizes=[50], d_attn=768, **kwargs):
        super().__init__(
            drop_cls=drop_cls, hidden_sizes=hidden_sizes, d_attn=d_attn, **kwargs
        )
        self.classifier = ClassifierModule(
            input_len=d_attn * 2,
            hidden_sizes=hidden_sizes,
            dropout=drop_cls,
        )

    def forward(
        self,
        data,
        select=["full_context", "judged_meaning"],
        mask=False,
        return_sim=False,
    ):
        if mask:
            x, mask_res = self.base_model(data, select, mask=mask)
        elif return_sim:
            return self.base_model(data, select, mask=mask, return_sim=True)
        else:
            x = self.base_model(data, select, mask=mask)
        x = torch.cat([x.max(dim=1)[0], x.mean(dim=1)], dim=-1)
        y = self.classifier(x)
        if mask:
            return mask_res, y
        return y


class PretrainedDXAModel(ModuleWrapper):
    def __init__(self, d_attn=768, drop_cls=0.3, hidden_sizes=[50], **kwargs):
        super().__init__(d_attn=d_attn, **kwargs)
        vocab_size = self.base_model.get_vocab_size()
        self.classifier = ClassifierModule(
            input_len=d_attn,
            output_len=vocab_size,
            hidden_sizes=hidden_sizes,
            dropout=drop_cls,
        )

    def forward(self, data, select=["full_context", "judged_meaning"], mask=True):
        x, mask_res = self.base_model(
            data, mask=mask, select=select
        )  # [batch, q_seq, d_embed] x [batch, k_seq, d_embed].T x [batch, k_seq, d_embed] -> [16, q_seq, d_attn]
        mask_inds = mask_res["mask_inds"]
        x = x[
            torch.arange(x.shape[0]), mask_inds, :
        ]  # [batch, q_seq, d_attn] -> [16, 1, d_attn]
        y = self.classifier(x)  # [16, 1, d_attn] -> [16, class_out]
        return y, mask_res


class SynonymModel(DXAModel):
    """GLiNER-based model, using WordNet senses
    as the equivalent of entity tokens. Otherwise,
    identical to the DXAModel implementation of GLiNER
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-cased",
        max_length: int = 512,
        d_attn: int = 768,
        n_syns: int = 4,
        dropout_p: float = 0.4,
        device="cpu",
    ):
        super().__init__(model_name, max_length, d_attn, dropout_p, device)
        import nltk

        nltk.download("wordnet")
        self.n_syns = n_syns
        self.syn_length = n_syns * 2
        self.def_length = max_length // 3
        self.ctx_length = (
            max_length - self.syn_length - self.def_length - 1
        )  # max_length-syn_length - def_length - misc
        self.wd_length = 1
        self.max_length = max_length

        # adding trainable token
        special_tokens = {"additional_special_tokens": ["<SYN>", "<MISC>"]}

        self.model.tokenizer.add_special_tokens(special_tokens)
        self.model.model.resize_token_embeddings(len(self.model.tokenizer))

    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.3,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ) -> torch.Tensor:
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
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        targ = torch.argmax(attn_weight[:, :, -1], dim=1)
        attn_weight = torch.nn.functional.dropout(
            attn_weight, dropout_p, training=self.training
        )
        return attn_weight @ value, targ, attn_weight

    def forward(
        self,
        data: Dict,
        select=["full_context", "homonym", "judged_meaning"],
        mask=False,
        return_sim=False,
    ):
        """largely borrowed from DXAModule, w/ minor changes to account for change
        in model
        """
        batch_size = len(data[select[1]])
        # special token processing
        data["sep"] = [self.model.tokenizer.sep_token] * batch_size
        syn, misc = self.model.tokenizer.additional_special_tokens
        data["syn"] = [[syn for _ in range(self.n_syns)] for _ in range(batch_size)]
        data["misc"] = [misc for _ in range(batch_size)]

        misc_toks = self.model.tokenizer(
            data["misc"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).to(self.device)

        sep_toks = self.model.tokenizer(
            data["sep"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).to(self.device)

        sep_toks["attention_mask"] = torch.zeros(sep_toks["attention_mask"].shape).to(
            self.device
        )
        sep_size = len(sep_toks["input_ids"][0])

        # context tokenization
        context_toks = self.model.tokenizer(
            data[select[0]],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.ctx_length - sep_size,
            add_special_tokens=False,
        ).to(self.device)

        syn_toks = {"input_ids": [], "attention_mask": []}
        syn_tags = {"input_ids": [], "attention_mask": []}
        for word, tags in zip(data[select[1]], data["syn"]):
            word_syns = self.wordnet_synonyms(word)
            if word_syns:
                # synonym tokenization
                syn_ids = self.model.tokenizer(
                    word_syns,
                    return_tensors="pt",
                    padding=True,
                    max_length=1,
                    truncation=True,
                    add_special_tokens=False,
                ).to(self.device)
                # synonym tag tokenization
                tag_ids = self.model.tokenizer(
                    tags[: len(word_syns)],
                    return_tensors="pt",
                    padding=False,
                    is_split_into_words=True,
                    add_special_tokens=False,
                ).to(self.device)

            else:
                syn_ids = {
                    "input_ids": torch.Tensor().to(self.device),
                    "attention_mask": torch.Tensor().to(self.device),
                }
                tag_ids = {
                    "input_ids": torch.Tensor().to(self.device),
                    "attention_mask": torch.Tensor().to(self.device),
                }

            syn_toks["input_ids"].append(syn_ids["input_ids"].flatten())
            syn_toks["attention_mask"].append(syn_ids["attention_mask"].flatten())
            syn_tags["input_ids"].append(tag_ids["input_ids"].flatten())
            syn_tags["attention_mask"].append(tag_ids["attention_mask"].flatten())

        def_toks = self.model.tokenizer(
            data[select[2]],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.def_length,
        ).to(self.device)

        # TODO - revisit the efficacy of masked LM as pre-training task for SER
        # if mask:
        #     syn_toks, masks = self.model.mask(synonyms, syn_toks)

        # interleave [SYN] tags and syns
        interleaved_syns = {"input_ids": [], "attention_mask": []}
        for key in ["input_ids", "attention_mask"]:
            for tag_inst, syn_inst in zip(syn_tags[key], syn_toks[key]):
                raw_inst = torch.stack([tag_inst, syn_inst]).T.flatten()
                padded_inst = torch.cat(
                    [
                        raw_inst,
                        torch.Tensor(
                            [
                                self.model.tokenizer.pad_token_id
                                for _ in range(self.syn_length - raw_inst.shape[-1])
                            ]
                        ).to(self.device),
                    ]
                )
                interleaved_syns[key].append(padded_inst)

        interleaved_syns["input_ids"] = torch.stack(
            interleaved_syns["input_ids"]
        ).long()
        interleaved_syns["attention_mask"] = torch.stack(
            interleaved_syns["attention_mask"]
        )

        # cat the tokenizations into single sequence
        compiled_input = {
            key: [
                torch.concat([il_syns, misc, sep, ctx.flatten(), df])
                for il_syns, misc, sep, ctx, df in zip(
                    interleaved_syns[key],
                    misc_toks[key],
                    sep_toks[key],
                    context_toks[key],
                    def_toks[key],
                )
            ]
            for key in ["input_ids", "attention_mask"]
        }
        input_seq = {
            "input_ids": torch.stack(compiled_input["input_ids"]).long(),
            "attention_mask": torch.stack(compiled_input["attention_mask"]).long(),
        }

        # contextembed forward
        input_embeds = self.model(input_seq, tokenize=False)

        # capture synonym tag (+ MISC) indices
        syn_inds = torch.arange(0, self.syn_length, step=2)
        syn_inds = torch.concat([syn_inds, torch.Tensor([self.syn_length])])

        # capture delimiters for pre-sep, post-sep ctx, and final definition
        sep_inds = (
            prectx_len := interleaved_syns["input_ids"].shape[-1] + 1,
            prectx_len + context_toks["input_ids"].shape[-1],
        )

        # accumulate embeddings + stack into tensor, pooling definition embeds
        seploc, ctxend = sep_inds
        context_embeds = torch.cat(
            [
                input_embeds[:, seploc:ctxend, :],
                torch.mean(input_embeds[:, ctxend:, :], dim=1).unsqueeze(1),
            ],
            dim=1,
        )
        synonym_embeds = input_embeds[:, syn_inds.long(), :]

        # attention projections
        refined_ctx = self.K(context_embeds)
        refined_syns = self.Q(synonym_embeds)
        aggre_value = self.V(context_embeds)

        # build attention mask
        attn_mask = torch.bmm(
            torch.cat(
                [
                    interleaved_syns["attention_mask"][:, syn_inds[:-1].long()],
                    torch.ones(batch_size, 1).to(self.device),
                ],
                dim=1,
            ).unsqueeze(2),
            torch.cat(
                [
                    context_toks["attention_mask"],
                    torch.ones(batch_size, 1).to(self.device),
                ],
                dim=1,
            ).unsqueeze(1),
        )

        # run attention
        x, attn_targets, sims = self.scaled_dot_product_attention(
            refined_syns,
            refined_ctx,
            aggre_value,
            attn_mask=attn_mask,
            dropout_p=self.drop_attn,
        )
        if return_sim:
            res = sims[torch.arange(batch_size), attn_targets, :]
            return res.tolist(), res.shape
        return x[torch.arange(batch_size), attn_targets, :]

    def wordnet_synonyms(self, w: str) -> list:
        """given a word, retrieve the synsets and grab the first
        word within each (giving us synonyms).
        TODO lemmatize
        TODO is there a better indicator of representative word than idx=0
        NOTE we might need this to not be in the form of list[list[str]]"""
        synsets = wn.synonyms(w)
        syns = list({wd[0] for wd in synsets if wd})[: self.n_syns]
        return syns


class ScoredSynonymModel(ModuleWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        data,
        select=["full_context", "judged_meaning"],
        mask=False,
        return_sim=False,
    ):
        if mask:
            x, mask_res = self.base_model(data, select, mask=mask)
        elif return_sim:
            return self.base_model(data, select, return_sim=return_sim)
        else:
            x = self.base_model(data, select)

        y = self.classifier(x)

        if mask:
            return mask_res, y
        return y


class PretrainedSynonymModel(PretrainedDXAModel):
    def __init__(
        self,
        base_type=SynonymModel,
        base_name="google-bert/bert-base-cased",
        max_length=512,
        d_attn=768,
        hidden_sizes=[50],
        drop_attn=0.4,
        drop_cls=0.4,
    ):
        super().__init__()
        self.name = base_name
        self.base_model = base_type(base_name, max_length, d_attn, dropout_p=drop_attn)
        vocab_size = self.base_model.get_vocab_size()
        self.classifier = ClassifierModule(input_len=d_attn, output_len=vocab_size)
        self.train_mode = False
        self.drop_cls = drop_cls

    def forward(
        self,
        data: Dict,
        select=["full_context", "homonym", "judged_meaning"],
        mask=True,
    ):
        x, mask_res = self.base_model(
            data, mask=mask, select=select
        )  # [batch, q_seq, d_embed] x [batch, k_seq, d_embed].T x [batch, k_seq, d_embed] -> [16, q_seq, d_attn]
        if mask:
            mask_inds = mask_res["mask_inds"]
            x = x[
                torch.arange(x.shape[0]), mask_inds, :
            ]  # [batch, q_seq, d_attn] -> [16, 1, d_attn]
        else:
            x = x.max(dim=1)[0].unsqueeze(1)
        y = self.classifier(x, self.drop_cls)  # [16, 1, d_attn] -> [16, class_out]
        return y, mask_res


## these are for using sbert without explicit similarity metric (i.e feeding embeddings directly to ffn)
class NoSimSentenceModule(torch.nn.Module):
    # sentence embedding module without similarity
    def __init__(self, device, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.device = device
        self.sbert_model = SentenceTransformer(model_name, device=self.device)

    def forward(self, data: Dict):
        context_embeds = self.sbert_model.encode(
            list(data["full_context"]), convert_to_tensor=True
        ).to(self.device)
        example_embeds = self.sbert_model.encode(
            list(data["example_sentence"]), convert_to_tensor=True
        ).to(self.device)
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
        self.sentence_embed = NoSimSentenceModule(
            device=self.device, model_name="all-MiniLM-L6-v2"
        )
        self.scorer = NoSimScoreModule
