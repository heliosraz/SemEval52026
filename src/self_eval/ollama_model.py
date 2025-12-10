# SemEval2026
# Task 5 --- Self Evaluation Scripts
# OLLAMA implementation of self-evaluation preliminary
# =============================================================================|
import argparse
import os
import json
from tqdm import tqdm

from ollama import generate
from pydantic import BaseModel, Field

from prompts import nl_template, sys_prompt_one
# =============================================================================|
# Data Model
class PromptResponse(BaseModel):
    thought: str
    reasoning: str
    score: int = Field(..., ge=1, le=5, description="prediction from 1-5")

fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": "score_prediction",
                "schema": PromptResponse.model_json_schema(),
                "strict": True
            }
}


# =============================================================================|
# Generation loop
def construct_response_function(**params) -> callable:
    """
    Closure encapsulating model parameters
    for iteration
    """
    def get_response(prompt: str):
        """Wrapper around ollama.generate to get a single response """
        return generate(model=params["model"],
                        prompt=prompt,
                        format=PromptResponse.model_json_schema(),
                        suffix=params["suffix"],
                        think=params["think"],
                        system=params["system_prompt"])

    return get_response


def generate_responses(instances, response_fn: callable):
    """
    full Response generator
    """
    for inst in instances:
        r = response_fn(inst)
        PromptResponse.model_validate_json(r.response)
        yield r.response


# =============================================================================|
# Main
def load_data(fp: str):
    with open(fp, 'r') as f:
        data = json.load(f)
        for item in list(data.items()):
            yield item


def main(args: dict):
    """"""
    prompts = (nl_template(d[1]["precontext"],
                            d[1]["sentence"],
                            d[1]["ending"],
                            d[1]["homonym"],
                            d[1]["judged_meaning"])
               for d in load_data(args["fp"]))

    inference_fn = construct_response_function(**args)

    with open(args["out"], 'w') as o:
        for doc in tqdm(iter(generate_responses(prompts, inference_fn))):
            json.dump(doc, o)
            o.write("\n")


# TODO logprobs + P(IK) head
# TODO argparse / config implementation
def parse_runtime_arguments(input_fp: str) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        required=True)
    parser.add_argument("--system_prompt")
    parser.add_argument("--think",
                        default=False)
    parser.add_argument("--suffix",
                        default="")
    parser.add_argument("-i", "--input_fp",
                        required=True)
    parser.add_argument("-o", "--output_fp",
                        required=True)


if __name__ == "__main__":
    clargs = {"model": "deepseek-r1",
              "system_prompt": sys_prompt_one(),
              "think": False,
              "suffix": "",
              "fp": "data/sample_data.json",
              "out": "results/deepseek.self_eval.prompt1.nl_template.run4.jsonl"
              }
    main(clargs)
