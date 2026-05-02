import pandas as pd
import os
import itertools
import torch
import yaml

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
    return torch.exp(
        -0.5
        * (
            (torch.arange(n_labels).unsqueeze(0) - mean.unsqueeze(1))
            / stdev.unsqueeze(1)
        )
        ** 2
    ).float()


def ft_data(path):
    split, _ = path.split("/")[-1].split(".")
    data = load_data(path)
    data["full_context"] = (
        data["precontext"] + " " + data["sentence"] + " " + data["ending"]
    )

    aug_data = {
        "target": [],
        "source": [],
        "homonym": [],
        "stdev": [],
        "average": [],
        "probs": [],
        "interval": [],
    }
    for _, row in data.iterrows():
        aug_data["target"] += [row["judged_meaning"]]
        aug_data["source"] += [row["full_context"]]
        aug_data["homonym"] += [row["homonym"]]
        aug_data["stdev"] += [row["stdev"]]
        aug_data["average"] += [row["average"] - 1]
        stdev = torch.Tensor([row["stdev"]])
        average = torch.Tensor([row["average"] - 1])
        stdev.masked_fill_(stdev == 0, float("1e-40"))
        aug_data["probs"] += sample_distribution(average, stdev).tolist()
        aug_data["interval"] += [
            (row["average"] - row["stdev"], row["average"] + row["stdev"])
        ]
    aug_data = pd.DataFrame(aug_data)
    aug_data.to_json(
        os.path.join(root, "{}_ft.json".format(split)),
        orient="index",
        indent=4,
    )


def mlm_data(path):
    split, _ = path.split("/")[-1].split(".")
    data = load_data(path)
    data["full_context"] = (
        data["precontext"] + " " + data["sentence"] + " " + data["ending"]
    )

    aug_data = {
        "target": [],
        "source": [],
        "homonym": [],
        "stdev": [],
        "average": [],
        "probs": [],
        "interval": [],
    }
    for _, row in data.iterrows():
        for target, reference in itertools.product(
            [row["judged_meaning"]], [row["full_context"], row["ending"]]
        ):
            if target and reference:
                aug_data["target"] += [target, reference]
                aug_data["source"] += [reference, reference]
                aug_data["stdev"] += [row["stdev"], 0]
                aug_data["homonym"] += [row["homonym"], row["homonym"]]
                aug_data["average"] += [row["average"] - 1, 4]
                stdev = torch.Tensor([row["stdev"], 0])
                average = torch.Tensor([row["average"], 5]) - 1
                stdev.masked_fill_(stdev == 0, float("1e-40"))
                aug_data["probs"] += sample_distribution(average, stdev).tolist()
                aug_data["interval"] += [
                    (row["average"] - row["stdev"], row["average"] + row["stdev"]),
                    (4, 4),
                ]
            aug_data["target"] += [row["judged_meaning"]]
            aug_data["source"] += [row["example_sentence"]]
            aug_data["homonym"] += [row["homonym"]]
            aug_data["stdev"] += [0]
            aug_data["average"] += [4]
            stdev = torch.Tensor([0])
            average = torch.Tensor([4])
            stdev.masked_fill_(stdev == 0, float("1e-40"))
            aug_data["probs"] += sample_distribution(average, stdev).tolist()
            aug_data["interval"] += [(4, 4)]
    aug_data = pd.DataFrame(aug_data)
    aug_data.to_json(
        os.path.join(root, "{}_mlm.json".format(split)),
        orient="index",
        indent=4,
    )


def add_context(path):
    df = load_data(path)
    df["full_context"] = df["precontext"] + " " + df["sentence"] + " " + df["ending"]
    df = df.drop("index", axis=1)
    df.to_json(path, orient="index", indent=4)


def read_yaml_file(file_path):
    try:
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)
        return config_data
    except yaml.YAMLError as exc:
        print(f"Error reading YAML file: {exc}")
        return None
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None


if __name__ == "__main__":
    root = os.path.join(".", "data")
    for f_name in ["test.json"]:
        add_context(os.path.join(root, f_name))
        # ft_data(os.path.join(root, f_name))
        # mlm_data(os.path.join(root, f_name))
#
