#!/usr/bin/env python3
# SemEval2026 --- Task 5
# OLLAMA implementation of self-evaluation preliminary
# ===========================================================================|
# Import Statements
import argparse
import json
from tqdm import tqdm

from ollama import generate
from pydantic import BaseModel, Field

from prompts import structured_template, nl_template, base_sys_prompt


# ===========================================================================|
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


# ===========================================================================|
# Generation loop
def construct_response_function(model: str, **params) -> callable:
    """
    Closure encapsulating model parameters
    for iteration
    """
    def get_response(prompt: str):
        """Wrapper around ollama.generate to get a single response """
        return generate(model=model,
                        prompt=prompt,
                        format=PromptResponse.model_json_schema(),
                        suffix=params["suffix"],
                        think=params["think"],
                        system=params["system_prompt"])

    return get_response


def generate_responses(instances: list[str], response_fn: callable):
    """
    full Response generator
    """
    for inst in instances:
        r = response_fn(inst)
        PromptResponse.model_validate_json(r.response)
        yield r.response


# ===========================================================================|
# Main
def load_data(fp: str):
    with open(fp, 'r') as f:
        yield from json.load(f)



def main(args: dict):
    """"""
    model = args["model"]
    for i, template in enumerate([nl_template, structured_template]):
        prompts = (template(d[1]["precontext"],
                            d[1]["sentence"],
                            d[1]["ending"],
                            d[1]["homonym"],
                            d[1]["judged_meaning"])

                   for d in load_data(args["input_fp"]))
        inference_fn = construct_response_function(model, **args)

        with open(f"{model}.{i}.jsonl", 'w') as o:
            for doc in tqdm(iter(generate_responses(prompts, inference_fn))):
                obj = json.loads(doc)
                json.dump(obj, o)
                o.write("\n")


def parse_runtime_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--think",
                        default=False)
    parser.add_argument("--suffix",
                        default="")
    parser.add_argument("-i", "--input_fp",
                        required=True),
    parser.add_argument("-m", "--model",
                        required=True)
    return vars(parser.parse_args())


if __name__ == "__main__":
    clargs = parse_runtime_arguments()
    clargs["system_prompt"] = base_sys_prompt()
    main(clargs)
