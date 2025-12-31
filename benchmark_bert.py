from transformers import AutoTokenizer, AutoModel
from load_data import make_dataset
import torch
import tqdm
from sys import argv, exit


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

if __name__ == "__main__":
    model_name = "google-bert/bert-base-uncased"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    ## Data Processing
    if len(argv)<1:
        print("No data file was provided")
        exit()
    else:
        data = make_dataset(argv[1])
    
    context_toks = tokenize(tokenizer, data['context']).to(device)
    example_toks = tokenize(tokenizer, data['example_sentence']).to(device)
    # get list of target offsets for each data instance for both context and example
    context_offsets, example_offsets = get_all_offsets(context_toks, example_toks, data)
    # remove offsets from both tokenized datasets
    context_toks.pop('offset_mapping', None)
    example_toks.pop('offset_mapping', None)
    with torch.no_grad():
        # model outputs = ** inputs for both context and example
        # get last hidden states, then use offsets to select target embeddings
        # cosine sim
        context_outputs = model(**context_toks)
        example_outputs = model(**example_toks)
    # dim = 0 feels weird but it has to do w/ shape of resultant embeddings
    # must select index i for batch dimension and offset[i] for token dimension
    cos = torch.nn.CosineSimilarity(dim=0)
    """
    context_tensors = obtain_final_embeddings(context_outputs, context_offsets)
    example_tensors = obtain_final_embeddings(example_outputs, example_offsets)
    similarities = cos(context_tensors, example_tensors)
    print(similarities[0])
    with open('results.json', 'w') as f:
        f.write(similarities)
    """
    sim = []
    for i in tqdm.tqdm(range(len(context_offsets))):
        context_tensor = context_outputs.last_hidden_state[i, context_offsets[i],:]
        example_tensor = example_outputs.last_hidden_state[i, example_offsets[i],:]
        sim.append(cos(context_tensor,example_tensor) * 5)
    sim = torch.array()
    print(sim[0])
    with open('results.txt', 'w') as f:
        f.writelines(sim)

    




    

