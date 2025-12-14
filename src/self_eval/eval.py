#!/usr/bin/env python3
import json
from scipy.stats import spearmanr

def acc_target(pred: int, avg: float, std_dev: float) -> int:
    """Evaluation Target #1: Accuracy within stdev

    if the model's output is within one standard deviation of the average,
    we consider that output to be correct wrt the task of sense confidence.
    """
    if (max(pred, avg) - min(pred, avg)) < std_dev:
        return 1
    return 0

def spearman_corr(pred: int, golds: list[int]) -> float:
    """Evaluation Target #2: Spearman Correlation Coefficient
    TODO describe metric
    """
    return spearmanr([pred] + golds)


def gold_parse(gd: dict) -> (list[int],float, float):
    """Retrieves significant gold values"""
    return gd["choices"], gd["average"], gd["stdev"]

def pred_parse(pd: dict) -> int:
    """Retrieves model output value"""
    return pd["score"]

def process_predictions(gold_fp: str, pred_fp: str) -> (list(float), float, float):
    """
    Process a directory of model output against a set of golds

    - param : gold_fp :: filepath containing data of the schema seen in semeval problem statement
    - param : pred_fp :: filepath containing model output of schema seen in README.md
    """
    o = []

    with open(gold_fp, 'r') as g, open(pred_fp, 'r') as p:
        gold_data, pred_data = list(json.load(g).items()), list(json.load(p).items())
        gold_items, pred_items = [gold_parse(d) for d in gold_data], [pred_parse(d) for d in pred_data]
        for (gr, avg, stdv), pred in zip(gold_items, pred_items):
            acc = acc_target(pred_items, avg, stdv)
            spm = spearman_corr(pred, gr)
            o.append((acc, spm))

    avgacc = avg([x[0] for x in o])
    avgspm = avg([x[1] for x in o])

    return o, avgacc, avgspm


# Output
def write_to_disk(scores: list[(int, float)], out_fp):
    with open(out_fp, 'w') as o:
        json.dump({i: {"Acc": x[0], "Spearman": x[1]} for i, x in enumerate(scores)},o)


def generate_table(scores: list[(int, float)]):
    pass

# Main
def main(params):
    individual_scores, avg_acc, avg_spearman = process_predictions(params["golds"],
                                                                   params["preds"])
    write_to_disk(individual_scores)
    generate_table(individual_scores)
    print(f"Average Accuracy = {avg_acc},\n Average Spearman Correlation: {avg_spearman}")

if __name__ == "__main__":
    args = {"golds": "data/sample_data.json",
            "preds": "results/model_output/ds.self_eval.run1.jsonl"}
    main(args)
