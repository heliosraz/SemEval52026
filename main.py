import json
from tqdm.asyncio import tqdm
from openai import OpenAI
from pydantic import BaseModel, Field

SYSTEM_PROMPT = ''' 
You are a helpful assistant. Given a pre-context and a target sentence, please provide a score out of a five point scale for how likely the story will have the given ending. You must provide your answer using JSON Schema with the following structure: thought, reasoning, and score (1-5).
'''

class ScorePrediction(BaseModel):
    thought: str
    reasoning: str
    score: int = Field(..., ge=1, le=5, description="Prediction score from 1 to 5")
    
format = {
            "type": "json_schema", 
            "json_schema": {
                "name": "score_prediction",
                "schema": ScorePrediction.model_json_schema(),
                "strict": True}
        }

def template(precontext, target, ending):
    return f'''
PRE-CONTEXT: {precontext}
TARGET SENTENCE: {target}
ENDING: {ending}
'''

def load_data(path: str):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)
status = 0

async def run_async(instance, client):
    prompt = template(instance[1]['precontext'],
                    instance[1]['sentence'],
                    instance[1]['ending'])
    
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=format
    )
    
    result = json.loads(response.choices[0].message.content)
    return {"id": instance[0], "prediction": result["score"]}

async def process_batch(instances, batch_size=32):
    results = []
    
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        tasks = [run_async(inst, client) for inst in batch]
        batch_results = []
        for f in tqdm.as_completed(tasks, total=len(tasks), desc="Processing tasks"):
            result = await f
            batch_results.append(result)
        results.extend(batch_results)
    
    return results

if __name__ == "__main__":
    data = load_data("./semeval26-05-scripts/data/dev.json")
    instances = list(data.items())
    
    results = asyncio.run(process_batch(instances, batch_size=32))
    with open("results/predictions.jsonl", "w") as fp:
        for r in tqdm(results):
            json.dump(r, fp)
            fp.write("\n")
        
