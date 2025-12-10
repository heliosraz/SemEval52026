
import asyncio
import json
from tqdm.asyncio import tqdm
from openai import OpenAI, AsyncOpenAI
from vllm import LLM
from src.model import fmt, template, ScorePrediction


def load_data(path: str):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)
status = 0


async def run_async(instance, client, **params):
    prompt = template(instance[1]['precontext'],
                      instance[1]['sentence'],
                      instance[1]['ending'])

    response = await client.chat.completions.create(
        model=params["model"],
        messages=[
            {"role": "system", "content": params["prompt"]},
            {"role": "user", "content": prompt}
        ],
        response_format=fmt
    )

    result = json.loads(response.choices[0].message.content)
    return {"id": instance[0], "prediction": result["score"]}


async def process_batch(instances, **params):
    results = []
    batch_size = params["batch_size"]
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        tasks = [run_async(inst, client, params) for inst in batch]
        batch_results = []
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing tasks"):
            result = await f
            batch_results.append(result)
        results.extend(batch_results)

    return results


def main(args: dict):
    data = load_data(args["filename"])
    instances = list(data.items())

    results = asyncio.run(process_batch(instances, batch_size=args["batch_size"]))

    with open("results/predictions.jsonl", "w") as f:
        for r in tqdm(results):
            json.dump(r, f)
            f.write("\n")


if __name__ == "__main__":
    runtime_args = {"filename": "data/sample_data.json",
                    "model": "Qwen/Qwen3-0.6B",
                    "batch_size": 32,
                    "prompt": SYSTEM_PROMPT}

    llm = LLM(model="facebook/opt-125m")
