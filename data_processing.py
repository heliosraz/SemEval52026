import pandas as pd
import os
import itertools
from datasets import Dataset, DatasetDict

# += sum([b-std<=a<=b+std for a, b, std in zip(
#                             torch.argmax(
#                                 y_pred,
#                                 dim = 1).flatten().float().tolist(),
#                             y_batch.flatten().float().tolist(),
#                             y_stdev.float().tolist())])
            
#             y_batch = torch.Tensor(batch["average"])-1
#             y_stdev = torch.Tensor(batch["stdev"])
#             y_stdev = y_stdev.masked_fill(y_stdev==0,1e-20)
            # y_probs = torch.exp(-0.5*((torch.arange(5).unsqueeze(0)-y_batch.unsqueeze(1))/y_stdev.unsqueeze(1))**2).float().to(device)
            # y_probs = y_probs / (y_probs.sum(dim=1, keepdim=True) + 1e-8)

def load_data(path):
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    new_data = data.reset_index()
    return new_data

def augment_data(path):
    split, _ = path.split("/")[-1].split(".")
    data = load_data(path)
    aug_data = {"target": [], "source": [], "stdev": [], "average": []}
    for _, row in data.iterrows():
        for target, reference in itertools.product(
                                                [
                                                    row["judged_meaning"],
                                                    row["example_sentence"]
                                                    ],
                                                [
                                                    row["context"],
                                                    row["sentence"],
                                                    row["ending"]
                                                    ]
                                            ):
            if target and reference:
                aug_data["target"] += [target, target, reference]
                aug_data["source"] += [reference, target, reference]
                aug_data["stdev"] += [row["stdev"], 0, 0]
                aug_data["average"] += [row["average"], 5, 5]
    aug_data = pd.DataFrame(aug_data)
    aug_data.to_json(
        os.path.join(root, "{}_augmented.json".format(split)),
        orient="index",
        indent = 4)
    
def mask_data(path):
    data = load_data(path)
    aug_data = {"target": [], "source": [], "mask": []}
    for _, row in data.iterrows():
        aug_data["target"] += [
                            row["judged_meaning"],
                            row["example_sentence"],
                            row["context"],
                            row["sentence"],
                            row["ending"]
                            ]
        aug_data["masked_sentence"] += ...
        
    
if __name__ == "__main__":
    root = os.path.join(".","data")
    for f_name in ["train.json", "dev.json"]:
        augment_data(os.path.join(root, f_name))
# 