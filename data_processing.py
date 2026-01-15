import pandas as pd
import os
import itertools
#from pathlib import Path
#from datasets import Dataset, DatasetDict

def load_data(path):
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    new_data = data.reset_index()
    return new_data

def augment_data(path):
    directory = "/".join(path.split("/")[:-1])
    name, _ = path.split("/")[-1].split(".")
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    aug_data = {"target": [], "reference": [], "stdev": [], "average": []}
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
                aug_data["reference"] += [reference, target, reference]
                aug_data["stdev"] += [row["stdev"], 0, 0]
                aug_data["average"] += [row["average"], 5, 5]
    aug_data = pd.DataFrame(aug_data)
    aug_data.to_json(
        os.path.join(directory, "{}_augmented.json".format(name)),
        orient="index",
        indent = 4)
    
if __name__ == "__main__":
    root = os.path.join(".","data")
    for f_name in ["train.json", "dev.json"]:
        augment_data(os.path.join(root, f_name))
# 