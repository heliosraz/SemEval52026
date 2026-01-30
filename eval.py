from data_structs import WordSenseData, AugWordSenseData
import models
import metrics
from sys import argv
from safetensors.torch import load_model
import torch
from data_processing import load_data, read_yaml_file
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import json
import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("DEVICE: {}".format(device))

task_dataset = {
    "eval": WordSenseData,
    "classifier-ft": AugWordSenseData,
    "pretrain": AugWordSenseData,
}
model_key = {
    "GeneralistModel_nosep": models.GeneralistModel_nosep,
    "GeneralistModel": models.GeneralistModel,
    "PretrainedGeneralistModel": models.PretrainedGeneralistModel,
    "BaselineModule": models.BaselineModule,
    "CrossContextSimilarityModule": models.CrossContextSimilarityModule,
}
metric_key = {
    "mask": metrics.accuracy,
    "average": metrics.range,
}
loss_key = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
}


def eval(model, data, select=["full_context", "judged_meaning"]):
    loader = DataLoader(
        data,
        batch_size=64,
    )
    res = pd.DataFrame(columns=["id", "prediction"])
    torch.use_deterministic_algorithms(True)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            preds = model(batch, select)
            if len(preds.shape) > 1:
                preds = (torch.argmax(preds, dim=1) + 1).cpu()
                # preds = torch.softmax(preds, dim=1)
                # preds = [
                #     sum([(i + 1) * prob for i, prob in enumerate(pred.tolist())])
                #     for pred in preds
                # ]
            else:
                # preds = torch.round(preds) + 1
                preds = (preds + 1).cpu()
            y = pd.DataFrame(preds, columns=["prediction"])
            y["id"] = batch["index"]
            res = pd.concat([res, y])
    res["id"] = res["id"].astype("str")
    res["prediction"] = res["prediction"]
    return res


def main(config):
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
            drop_attn=config["model"]["drop_attn"],
            drop_cls=config["model"]["drop_cls"],
        ).to(device)
    else:
        model = base_model(
            model_name=encoder, max_length=config["model"]["max_len"]
        ).to(device)
    if not config["model"]["huggingface"]:
        load_model(model, config["evaluation"]["prev_path"])

    # Data Processing
    test_df = load_data(config["data"]["data"])
    test_set = task_dataset[task](test_df)

    # RUN
    result = eval(model, test_set, config["data"]["input_tags"])

    # Saving
    result_json = result.to_json(orient="records", lines=True)

    outdir = os.path.join(config["data"]["output"], config["model"]["name"])
    os.makedirs(outdir, exist_ok=True)
    fp = os.path.join(outdir, "predictions.jsonl")
    with open(fp, "w") as f:
        f.write(result_json)


if __name__ == "__main__":
    config = read_yaml_file(argv[1])
    main(config)
