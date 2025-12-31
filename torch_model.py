import torch
from transformers import AutoModel, AutoTokenizer
from benchmark_bert import get_all_offsets
import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class SimModel(torch.nn.Module):

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

class BertModel(torch.nn.Module):
    # torch no grad should 

    def __init__(self):
        super().__init__()
        self.bert_layer = AutoModel.from_pretrained("google-bert/bert-base-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        self.cosine = torch.nn.CosineSimilarity(dim=0)

    def forward(self, data):
        # taken from benchmark_bert.py
        context_toks = self.tokenizer(list(data['context']), return_tensors='pt', padding=True, return_offsets_mapping=True).to(device)
        example_toks = self.tokenizer(list(data['example_sentence']), return_tensors='pt', padding=True, return_offsets_mapping=True).to(device)
        # get list of target offsets for each data instance for both context and example
        context_offsets, example_offsets = get_all_offsets(context_toks, example_toks, data)
        # remove offsets from both tokenized datasets
        context_toks.pop('offset_mapping', None)
        example_toks.pop('offset_mapping', None)
        context_outputs = self.bert_layer(**context_toks)
        example_outputs = self.bert_layer(**example_toks)
        sim = []
        for i in tqdm.tqdm(range(len(context_offsets))):
            context_tensor = context_outputs.last_hidden_state[i, context_offsets[i],:]
            example_tensor = example_outputs.last_hidden_state[i, example_offsets[i],:]
            sim.append(self.cosine(context_tensor,example_tensor) * 5)
        return sim


