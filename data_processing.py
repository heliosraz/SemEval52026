import pandas as pd
import os
import itertools
import torch

            
#             y_batch = torch.Tensor(batch["average"])-1
#             y_stdev = torch.Tensor(batch["stdev"])
#             y_stdev = y_stdev.masked_fill(y_stdev==0,1e-20)
            # y_probs = torch.exp(-0.5*((torch.arange(5).unsqueeze(0)-y_batch.unsqueeze(1))/y_stdev.unsqueeze(1))**2).float().to(device)
            # y_probs = y_probs / (y_probs.sum(dim=1, keepdim=True) + 1e-8)

def load_data(path):
    data = pd.read_json(path).transpose()
    new_data = data.reset_index()
    return new_data

def sample_distribution(mean, stdev, n_labels=5):
    return torch.softmax(torch.exp(-0.5*((torch.arange(n_labels).unsqueeze(0)-mean.unsqueeze(1))/stdev.unsqueeze(1))**2),dim=-1).float()

def augment_data(path):
    split, _ = path.split("/")[-1].split(".")
    data = load_data(path)
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    
    aug_data = {"target": [], "source": [], "stdev": [], "average": [], "probs": [], "interval": []}
    for _, row in data.iterrows():
        for target, reference in itertools.product(
                                                [
                                                    row["judged_meaning"],
                                                    row["example_sentence"]
                                                ],
                                                [
                                                    row["example_sentence"],
                                                    row["context"],
                                                    row["ending"]
                                                    ]
                                            ):
            if target and reference:
                aug_data["target"] += [target, reference]
                aug_data["source"] += [reference, reference]
                aug_data["stdev"] += [row["stdev"], 0]
                aug_data["average"] += [row["average"]-1, 4]
                stdev = torch.Tensor([row["stdev"], 0])
                average = torch.Tensor([row["average"], 5])-1
                stdev.masked_fill_(stdev==0, float("1e-20"))
                aug_data["probs"] += sample_distribution(stdev, average).tolist()
                aug_data["interval"] += [(row["average"]+row["stdev"], row["average"]-row["stdev"]), (4,4)]
    aug_data = pd.DataFrame(aug_data)
    aug_data.to_json(
        os.path.join(root, "{}_augmented.json".format(split)),
        orient="index",
        indent = 4)
    
def add_context(path):
    df = load_data(path)
    df['full_context'] = df['precontext'] + ' ' + df['sentence'] + ' ' + df['ending']
    df = df.drop("index", axis = 1)
    df.to_json(
        path,
        orient="index",
        indent = 4)
        
    
if __name__ == "__main__":
    root = os.path.join(".","data")
    for f_name in ["train.json", "dev.json"]:
        add_context(os.path.join(root, f_name))
        # augment_data(os.path.join(root, f_name))
# 