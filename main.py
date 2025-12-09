import json
from tqdm.asyncio import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field
from data_processing import load_system, load_data, prompt_template
import sys

import asyncio
from openai import AsyncOpenAI

class ScorePrediction(BaseModel):
    thought: str
    reasoning: str
    score: int = Field(..., ge=1, le=5, description="Prediction score from 1 to 5")
    
schema = {
            "type": "json_schema", 
            "json_schema": {
                "name": "score_prediction",
                "schema": ScorePrediction.model_json_schema(),
                "strict": True}
        }

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)
status = 0

async def run_async(instance, system_prompt, schema, client):
    prompt = prompt_template(instance[1]['precontext'],
                    instance[1]['sentence'],
                    instance[1]['ending'])
    
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format=schema
    )
    
    result = json.loads(response.choices[0].message.content)
    return {"id": instance[0], "prediction": result["score"]}

async def process_batch(batch, system_prompt):
    tasks = [run_async(inst, system_prompt, client) for inst in batch]
    batch_results = []
    for f in tqdm.as_completed(tasks, 
                            total=len(tasks), 
                            desc="Processing tasks"):
        result = await f
        batch_results.append(result)
    
    return batch_results

def main(instances, system_prompt, batch_size):
    results = []
    for i in range(0, len(instances), batch_size=32):
        batch = instances[i:i+batch_size]
        results.extend(asyncio.run(process_batch(batch, system_prompt)))
    return results

if __name__ == "__main__":
    if len(sys.argv)>1:
        data = load_data(sys.argv[1])
        prompt = load_system("data/system_prompt.jsonl", sys.argv[2])
    else:
        data = load_data("data/dev.json")
        prompt = load_system("data/system_prompt.jsonl", "1")
    instances = list(data.items())
    
    results = main(instances, prompt)
    with open("results/predictions.jsonl", "w") as fp:
        for r in tqdm(results):
            json.dump(r, fp)
            fp.write("\n")
        
