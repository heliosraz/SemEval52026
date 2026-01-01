#!/usr/bin/env python3
# SemEval2026
# Task 5 --- Self Evaluation Scripts
# vLLM implementation of self-evaluation script
# ----------------------------------------------------------------------------|
# Import Statements
import asyncio
import json
import os
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------|
# Data Model
class ScorePrediction(BaseModel):
    thought: str
    reasoning: str
    score: int = Field(..., ge=1, le=5, description="Predicted score from 1-5")


# /* Prompting Constants */
SYSTEM_PROMPT: str = """
You are a helpful assistant. Given a pre-context and a target sentence,
please provide a score out of a five point scale for how likely the story
will have the given ending. You must provide your answer using JSON Schema
with the following structure: thought, reasoning, and score (1-5).
"""

FMT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "score_prediction",
        "schema": ScorePrediction.model_json_schema(),
        "strict": True
    }
}


def template(p: str, t: str, e: str):
    return f'''
    PRE-CONTEXT: {p}
    TARGET SENTENCE: {t}
    ENDING: {e}
    '''


# ----------------------------------------------------------------------------|
# API Client
def init_client(client_type, **params):
    """"""
    if client_type == "async":
        return AsyncOpenAI(api_key=params["api_key"],
                           base_url=params["base_url"])
    else:
        return OpenAI(api_key=params["api_key"],
                      base_url=params["base_url"])


async def run_async(instance, client, **params):
    """"""
    prompt = template(instance[1]['precontext'],
                      instance[1]['sentence'],
                      instance[1]['ending'])

    response = await client.chat.completions.create(
        model=params['model'],
        messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                  {'role': 'user', 'content': prompt}],
        response_format=FMT
    )

    result = json.loads(response.choices[0].message.content)
    return {'id': instance[0], 'prediction': result['score']}


async def process_batch(instances, **params):
    """"""
    batch_size = params["batch_size"]
    client = init_client(params["client"])
    res = []

    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        tasks = [run_async(inst, client, params) for inst in batch]
        batch_results = []
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="processing..."):
            result = await f
            batch_results.append(result)
        res.extend(batch_results)

    return res


# ----------------------------------------------------------------------------|
# Driver Code
def load_data(fp: str | os.PathLike):
    with open(fp, 'r') as f:
        data = json.load(f)
    return data


def main(**kwargs):
    data = load_data(kwargs["input_fname"])
    instances = list(data.items())
    results = asyncio.run(process_batch(instances, kwargs))
    with open(kwargs["output_fname"], 'w') as fp:
        for r in tqdm(results):
            json.dump(r, fp)
            fp.write('\n')


if __name__ == "__main__":
    runtime_parameters = {"api_key": "EMPTY",
                          "base_url": "http://localhost:8000/v1",
                          "batch_size": 32,
                          "input_fname": "data/sample_data.json",
                          "output_fname": "results/predictions.jsonl",
                          "model": "Qwen/Qwen3-0.6B",
                          "client": "async"
                          }
    main(runtime_parameters)
