
# =============================================================================|
import os
from ollama import generate
import json

from tqdm import tqdm

from pydantic import BaseModel, Field
from argparse import Namespace

# =============================================================================|
SYSTEM_PROMPT = '''
You are a helpful assistant.
Given a pre-context and a target sentence,
please provide a score out of a five point
scale for how likely the story will have the given ending.
You must provide your answer using JSON Schema with the following structure:
thought, reasoning, and score (1-5).
'''


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


def template(precontext, target, ending):
    return f'''
        PRE-CONTEXT: {precontext}
        TARGET SENTENCE: {target}
        ENDING: {ending}
    '''


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
                        system=params["system_prompt"],
                        raw=True)

    return get_response


def generate_responses(instances, response_fn: callable):
    """
    full Response generator
    """
    for inst in instances:
        r = response_fn(inst)
        yield PromptResponse.model_validate_json(r.message.content)


# =============================================================================|
# Main
def load_data(fp: str):
    with open(fp, 'r') as f:
        for line in f.readlines():
            yield json.loads(line)


def main(args: dict):
    """"""
    prompts = (template(d[1]["precontext"], d[1]["sentence"], d[1]["ending"]) for x in load_data(args["fp"]))
    inference_fn = construct_response_function(**args)

    with open(args["out"], 'w') as o:
        for doc in tqdm(iter(generate_responses(prompts, inference_fn))):
            json.dump(o, doc)
            o.write("\n")


# TODO logprobs + P(IK) head
# TODO argparse / config implementation
if __name__ == "__main__":
    clargs = {"model": "deepseek-r1",
              "system_prompt": SYSTEM_PROMPT,
              "think": False,
              "suffix": "",
              "fp": "data/sample_data.json",
              "out": "results/deepseek.self_eval.prompt1.jsonl"
              }
    main(clargs)
