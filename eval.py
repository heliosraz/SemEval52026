from data_structs import WordSenseData, AugWordSenseData
import models
import metrics
from sys import argv
from safetensors.torch import save_file, load_file
import torch
from data_processing import load_data, read_yaml_file
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import json
import os
from tqdm import tqdm
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from nltk_tag_script import tok_span, tok_tag, load_tagset

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)

device = torch.device("mps")
print("DEVICE: {}".format(device))

task_dataset = {
    "eval": WordSenseData,
    "finetuning": WordSenseData,
    "pretrain": AugWordSenseData,
}
model_key = {
    "GeneralistModel_nosep": models.GeneralistModel_nosep,
    "GeneralistModel": models.GeneralistModel,
    "PretrainedGeneralistModel": models.PretrainedGeneralistModel,
    "BaselineModule": models.BaselineModule,
    "CrossContextSimilarityModule": models.CrossContextSimilarityModule,
    "SynonymModel": models.SynonymModel,
    "ScoredSynonymModel": models.ScoredSynonymModel,
    "PretrainedSynonymModel": models.PretrainedSynonymModel,
    "GeneralistModelScored": models.GeneralistModelScored,
}
metric_key = {
    "mask": metrics.accuracy,
    "average": metrics.range,
    "interval": metrics.range,
}
loss_key = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
}


def load_model(model, path):
    state_dict = load_file(path)
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {path} (strict=False)")


def gather_nltk(data: Dataset, select=["full_context", "judged_meaning"]):
    res = {i: {"span": [], "tag": []} for i in select}
    for ind in res:
        for txt in data[ind]:
            res[ind]["span"].append(tok_span(txt))
            res[ind]["tag"].append([tag[1] for tag in tok_tag(txt)])
    return res


def tokenize(model, data: Dataset, select=["full_context", "judged_meaning"]):
    res = {}
    for col in select:
        res[col] = model.tokenizer(
            data[col], return_offsets_mapping=True, add_special_tokens=False
        )
    return res


def match_pos(toks, tag_span):
    res = {col: [] for col in toks}
    for col in toks:
        offsets = toks[col].pop("offset_mapping")

        curr = tag_span[col]

        for (
            offset,
            tags,
            spans,
        ) in zip(offsets, curr["tag"], curr["span"]):
            curr_pos = []
            if spans == -1:
                curr_pos.append(tags[0])
            else:
                for off in offset:
                    if spans[0][0] > off[0] or off[1] > spans[0][1]:
                        tags.pop(0)
                        spans.pop(0)
                    curr_pos.append(tags[0])
            res[col].append(curr_pos)

    return res


def eval_sims(
    model,
    data,
    select=["full_context", "judged_meaning"],
    overwrite_tag=None,
    tagset=[],
):
    loader = DataLoader(data, batch_size=64, num_workers=0, shuffle=False)
    select_tagset = {
        st: tagset if st not in overwrite_tag else overwrite_tag[st] for st in select
    }
    conf_matrix = {
        tag1: {tag2: [] for tag2 in select_tagset[select[1]]}
        for tag1 in select_tagset[select[0]]
    }  # for each POS pair (set product) {key: query: list of scores}
    conf_mean = {
        tag1: {tag2: 0 for tag2 in select_tagset[select[1]]}
        for tag1 in select_tagset[select[0]]
    }
    conf_std = {
        tag1: {tag2: 0 for tag2 in select_tagset[select[1]]}
        for tag1 in select_tagset[select[0]]
    }
    conf_count = {
        tag1: {tag2: 0 for tag2 in select_tagset[select[1]]}
        for tag1 in select_tagset[select[0]]
    }

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            # match tag and spans to sims
            # match tag and spans to model tokenizer
            toks = tokenize(model, batch)

            tag_spans = gather_nltk(batch)
            for t in overwrite_tag:
                tag_spans[t]["tag"] = [
                    list(select_tagset[t]) for _ in tag_spans[t]["tag"]
                ]
                tag_spans[t]["span"] = [-1 for _ in tag_spans[t]["span"]]
            matched = match_pos(toks, tag_spans)
            pos_pairs = [
                list(product(pos_k, pos_q)) for pos_k, pos_q in zip(*matched.values())
            ]

            sims = model(batch, select, return_sim=True)
            for pairs in pos_pairs:
                for (pos_k, pos_q), sim in zip(pairs, sims):
                    if pos_k in conf_matrix and pos_q in conf_matrix[pos_k]:
                        conf_matrix[pos_k][pos_q].append(sim)
    for k, qs in conf_matrix.items():
        for q, scores in qs.items():
            conf_mean[k][q] = np.mean(scores) if scores else 0
            conf_std[k][q] = np.std(scores) if scores else 0
            conf_count[k][q] = len(scores) if scores else 0
    return conf_mean, conf_std, conf_count


def eval(model, data, select=["full_context", "judged_meaning"]):
    loader = DataLoader(data, batch_size=64, num_workers=0, shuffle=False)
    res = pd.DataFrame(columns=["id", "prediction"])
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            preds = model(batch, select)
            if len(preds.shape) > 1:
                # preds = (torch.argmax(preds, dim=1) + 1).cpu()
                preds = torch.softmax(preds, dim=1)
                preds = [
                    sum([(i + 1) * prob for i, prob in enumerate(pred.tolist())])
                    for pred in preds
                ]
            else:
                # preds = torch.round(preds) + 1
                preds = (preds + 1).cpu()
            y = pd.DataFrame(preds, columns=["prediction"])
            y["id"] = batch["index"]
            res = pd.concat([res, y])
    res["id"] = res["id"].astype("str")
    res["prediction"] = res["prediction"]
    return res


def show_heatmap(title, fname, data_dict: dict[str, dict[str, float]]):
    keys = [k for k in data_dict]
    query = [q for q in data_dict[keys[0]]]
    data = np.array([list(q.values()) for _, q in data_dict.items()])

    fig, ax = plt.subplots()
    im = ax.pcolorfast(data)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys, fontsize=8, va="center")

    ax.set_xticks(np.arange(len(query)))
    ax.set_xticklabels(
        query, rotation=45, ha="right", rotation_mode="anchor", fontsize=8
    )

    # # Loop over data dimensions and create text annotations.
    for i in range(len(keys)):
        for j in range(len(query)):
            text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")


def main(config):
    task = config["evaluation"]["task"]
    ## Model Initialization
    name = "{}.{}".format(
        model_key[config["model"]["architecture"]], config["model"]["encoder"]
    )
    base_model = model_key[config["model"]["architecture"]]
    encoder = config["model"]["encoder"]
    if config["model"]["wrapper"]:
        wrapper = model_key[config["model"]["wrapper"]]
        model = wrapper(
            base_type=base_model,
            base_name=encoder,
            max_length=config["model"]["max_len"],
            hidden_sizes=config["model"]["hidden_sizes"],
            d_attn=config["model"]["d_attn"],
            device=device,
        ).to(device)
    else:
        model = base_model(
            model_name=encoder,
            max_length=config["model"]["max_len"],
            device=device,
        ).to(device)
    if config["evaluation"]["prev_path"]:
        load_model(model, config["evaluation"]["prev_path"])

    overwrite_tag = config["data"]["eval_tags"]

    # Data Processing
    test_df = load_data(config["data"]["data"])
    test_set = task_dataset[task](test_df)
    test_tagset = load_tagset()

    # RUN
    # result = eval(model, test_set, config["data"]["input_tags"])
    means, stds, counts = eval_sims(
        model,
        test_set,
        config["data"]["input_tags"],
        tagset=test_tagset,
        overwrite_tag=overwrite_tag,
    )

    # display map
    show_heatmap(
        f"POS pair attention means for {config["model"]["name"]}",
        f"{config["model"]["name"]}_mean",
        means,
    )
    show_heatmap(
        f"POS pair attention stdev for {config["model"]["name"]}",
        f"{config["model"]["name"]}_stdev",
        stds,
    )
    show_heatmap(
        f"POS pair attention counts for {config["model"]["name"]}",
        f"{config["model"]["name"]}_counts",
        counts,
    )

    # Saving
    # result_json = result.to_json(orient="records", lines=True)

    outdir = os.path.join(config["data"]["output"], config["model"]["name"])
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, "predictions.jsonl")
    # with open(fp, "w") as f:
    #     f.write(result_json)


if __name__ == "__main__":
    config = read_yaml_file(argv[1])
    main(config)
