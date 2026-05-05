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
from nltk_tag_script import tok_span, tok_tag, load_tagset, tok_span_and_tag

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
    "DXAModel_nosep": models.DXAModel_nosep,
    "DXAModel": models.DXAModel,
    "PretrainedDXAModel": models.PretrainedDXAModel,
    "BaselineModule": models.BaselineModule,
    "CrossContextSimilarityModule": models.CrossContextSimilarityModule,
    "SynonymModel": models.SynonymModel,
    "ScoredSynonymModel": models.ScoredSynonymModel,
    "PretrainedSynonymModel": models.PretrainedSynonymModel,
    "ScoredDXAModel": models.ScoredDXAModel,
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
            try:
                spans, tags = tok_span_and_tag(txt)
                res[ind]["span"].append(spans)
                res[ind]["tag"].append(tags)
            except not tok_span(txt):
                print(f"empty span for {txt}")
    return res


def tokenize(model, data: Dataset, select=["full_context", "judged_meaning"]):
    res = {}
    for col in select:
        res[col] = model.tokenizer(
            data[col], return_offsets_mapping=True, add_special_tokens=False
        )
    return res


def match_pos(toks, tag_spans, shape, select={"full_context": 1, "judged_meaning": 0}):
    res = {col: [] for col in toks}
    for col in toks:
        offsets = toks[col].pop("offset_mapping")

        curr = tag_spans[col]

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
                    if off[0] > spans[0][1]:
                        tags.pop(0)
                        spans.pop(0)
                    if tags:
                        curr_pos.append(tags[0])
                if select[col] < len(shape):
                    curr_pos = curr_pos + [
                        "[PAD]" for _ in range(shape[select[col]] - len(offset))
                    ]
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
    conf_cv = {
        tag1: {tag2: 0 for tag2 in select_tagset[select[1]]}
        for tag1 in select_tagset[select[0]]
    }

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            # match tag and spans to sims
            # -> match tag and spans to model tokenizer
            toks = tokenize(model, batch, select=select[:2])
            # get token tags and spans from nltk
            tag_spans = gather_nltk(batch, select=select[:2])
            for t in overwrite_tag:
                tag_spans[t]["tag"] = [
                    list(select_tagset[t]) for _ in tag_spans[t]["tag"]
                ]
                tag_spans[t]["span"] = [-1 for _ in tag_spans[t]["span"]]

            # gather similarities
            sims, sim_shape = model(batch, select, return_sim=True)

            # match up the pos from of the text toks (model specifics)
            matched = match_pos(
                toks,
                tag_spans,
                shape=sim_shape,
                select={label: i + 1 for i, label in enumerate(select[1::-1])},
            )
            pos_pairs = [
                list(product(pos_q, pos_k)) for pos_k, pos_q in zip(*matched.values())
            ]

            for pairs, sim_inst in zip(pos_pairs, sims):
                for (pos_q, pos_k), sim in zip(pairs, sim_inst):
                    if pos_k in conf_matrix and pos_q in conf_matrix[pos_k]:
                        conf_matrix[pos_k][pos_q].append(sim)
    for k, qs in conf_matrix.items():
        for q, scores in qs.items():
            conf_mean[k][q] = np.mean(scores) if scores else 0
            conf_std[k][q] = np.std(scores) if scores else 0
            conf_count[k][q] = len(scores) if scores else 0
            conf_cv[k][q] = (
                conf_std[k][q] / conf_mean[k][q] if conf_mean[k][q] != 0 else 0
            )
    return conf_mean, conf_std, conf_count, conf_cv


def evaluate(model, data, select=["full_context", "judged_meaning"]):
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


def show_heatmap(
    title,
    data_dict: dict[str, dict[str, float]],
    select=["full_context", "judged_meaning"],
    fig=None,
    ax=None,
):
    y = sorted([k for k in data_dict])
    x = sorted([q for q in data_dict[y[0]]])
    if ax is None or fig is None:
        fig_w = max(4, len(x) * 0.8 + 2)
        fig_h = max(4, len(y) * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = np.array([[data_dict[y][x] for x in x] for y in y])

    im = ax.pcolorfast(data)

    ax.set_xticks(np.arange(len(x) + 1))  # major at edges
    ax.set_yticks(np.arange(len(y) + 1))  # major at edges
    ax.tick_params(which="major", length=0, labelbottom=False, labelleft=False)
    ax.grid(which="major", color="white", linewidth=0.5)

    ax.set_xticks(np.arange(len(x)) + 0.5, minor=True)  # minor at centers
    ax.set_yticks(np.arange(len(y)) + 0.5, minor=True)  # minor at centers
    ax.set_xticklabels(x, minor=True, rotation=90, fontsize=28, ha="center")
    ax.set_yticklabels(y, minor=True, fontsize=28, va="center")
    ax.tick_params(which="minor", length=0)

    ax.set_ylabel(select[0], fontsize=48)
    ax.set_xlabel(select[1], fontsize=36)

    ax.grid(which="major", color="white", linewidth=0.5)

    fig.colorbar(im, ax=ax, fraction=0.3, shrink=1, pad=0.1)
    ax.set_title(title, fontsize=48)
    return fig


def main(config, ranges):
    task = config["evaluation"]["task"]
    ## Model Initialization
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
    for low, high in ranges:
        test_df = load_data(config["data"]["data"])
        filtered_test = test_df[test_df["average"].between(low, high)]
        test_set = task_dataset[task](filtered_test)

        # test_set = task_dataset[task](test_df)
        test_tagset = load_tagset()

        # RUN
        # result = evaluate(model, test_set, config["data"]["input_tags"])
        means, stds, counts, cv = eval_sims(
            model=model,
            data=test_set,
            select=config["data"]["input_tags"],
            tagset=test_tagset,
            overwrite_tag=overwrite_tag,
        )

        # Saving
        # result_json = result.to_json(orient="records", lines=True)

        outdir = os.path.join(config["data"]["output"], config["model"]["name"])
        os.makedirs(outdir, exist_ok=True)
        fp = os.path.join(outdir, f"{config["model"]["name"]}_predictions.jsonl")
        # with open(fp, "w") as f:
        #     print("results dir: {}".format(fp))
        #     f.write(result_json)

        yield means, stds, counts, cv


if __name__ == "__main__":
    arches_out = {}
    input_tags = {}
    example_types = {
        "all": (float("-inf"), float("inf")),
        "negative": (0, 2.99),
        "neutral": (3, 3.99),
        "positive": (4, 5),
    }
    # gather data
    if len(argv) > 1:
        config = read_yaml_file(argv[1])
        arch, _ = config["model"]["name"].split("-")
        arch = arch.lower()
        means, stds, counts, cv = main(config)
        if arch not in arches_out:
            arches_out[arch] = {}
        arches_out[arch][config["model"]["name"]] = {
            "mean": means,
            "std": stds,
            "count": counts,
        }
        input_tags[arch] = config["data"]["input_tags"]

    else:
        import os

        for root, dirs, _ in os.walk("configs"):
            for dr in tqdm(dirs):
                arch, _ = dr.split("-")
                fp = os.path.join(root, dr, "eval.yaml")
                print(fp)
                if fp == "configs/bert-baseline/eval.yaml":
                    continue
                config = read_yaml_file(fp)
                for example_type, (means, stds, counts, cv) in zip(
                    example_types, main(config, example_types.values())
                ):
                    if arch not in arches_out:
                        arches_out[arch] = {}
                    if dr not in arches_out[arch]:
                        arches_out[arch][dr] = {}
                    arches_out[arch][dr][example_type] = {
                        "mean": means,
                        "std": stds,
                        "count": counts,
                        "cv": cv,
                    }
                    if arch not in input_tags:
                        input_tags[arch] = config["data"]["input_tags"]

    for i, (arch, bases) in enumerate(arches_out.items()):
        temp = list(bases.keys())[0]
        for example_type in example_types:
            # plt init
            query = sorted([q for q in bases[temp][example_type]["mean"]])
            keys = sorted([k for k in bases[temp][example_type]["mean"][query[0]]])
            fig_w = max(5, len(keys) * 0.8 + 2)
            fig_h = max(5, len(query) * 0.8 + 2)
            fig1, mean_axes = plt.subplots(
                1, len(bases), figsize=(fig_w * len(bases), fig_h)
            )
            fig2, std_axes = plt.subplots(
                1, len(bases), figsize=(fig_w * len(bases), fig_h)
            )
            fig3, count_axes = plt.subplots(
                1, len(bases), figsize=(fig_w * len(bases), fig_h)
            )

            fig4, cv_axes = plt.subplots(
                1, len(bases), figsize=(fig_w * len(bases), fig_h)
            )

            mean_axes = np.atleast_1d(mean_axes)
            std_axes = np.atleast_1d(std_axes)
            count_axes = np.atleast_1d(count_axes)
            cv_axes = np.atleast_1d(cv_axes)

            for j, model_n in enumerate(bases):
                show_heatmap(
                    model_n,
                    bases[model_n][example_type]["mean"],
                    input_tags[arch],
                    fig1,
                    mean_axes[j],
                )
                show_heatmap(
                    model_n,
                    bases[model_n][example_type]["std"],
                    input_tags[arch],
                    fig2,
                    std_axes[j],
                )
                show_heatmap(
                    model_n,
                    bases[model_n][example_type]["count"],
                    input_tags[arch],
                    fig3,
                    count_axes[j],
                )
                show_heatmap(
                    model_n,
                    bases[model_n][example_type]["cv"],
                    input_tags[arch],
                    fig4,
                    cv_axes[j],
                )

            key, query, *_ = input_tags[arch]
            fig1.suptitle(
                "Mean of <{}, {}> {} Example Pairs \n Similarity Scores for {}".format(
                    key, query, example_type.capitalize(), arch.upper()
                ),
                fontsize=60,
                y=1.02,
            )
            fig2.suptitle(
                "Standard Deviation of <{}, {}> {} Example Pairs \n Similarity Scores for {}".format(
                    key, query, example_type.capitalize(), arch.upper()
                ),
                fontsize=60,
                y=1.02,
            )
            fig3.suptitle(
                "Counts of <{}, {}> {} Example Pairs \n Similarity Scores for {}".format(
                    key, query, example_type.capitalize(), arch.upper()
                ),
                fontsize=60,
                y=1.02,
            )

            fig4.suptitle(
                "CV of <{}, {}> {} Example Pairs \n Similarity Scores for {}".format(
                    key, query, example_type.capitalize(), arch.upper()
                ),
                fontsize=60,
                y=1.02,
            )
            fig1.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            fig4.tight_layout()

            fig1.savefig(
                f"{arch}_mean_{example_type}.pdf", dpi=150, bbox_inches="tight"
            )
            fig2.savefig(f"{arch}_std_{example_type}.pdf", dpi=150, bbox_inches="tight")
            fig3.savefig(
                f"{arch}_count_{example_type}.pdf", dpi=150, bbox_inches="tight"
            )
            fig4.savefig(f"{arch}_cv_{example_type}.pdf", dpi=150, bbox_inches="tight")

            plt.close("all")
