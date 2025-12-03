import vllm
import json
from tqdm import tqdm
from openai import OpenAI
import re

# TODO: load data
# TODO: query model using openAI API
# TODO: save results

SYSTEM_PROMPT = ''' 
You are a helpful assistant. Given a pre-context and a target sentence, please provide a score out of a continuous 5 point scale for how likely the story will have the given ending. You must provide you answer at the end of the response as so: SCORE: [score]
'''
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def template(precontext, target, ending):
    return f'''
PRE-CONTEXT: {precontext}
TARGET SENTENCE: {target}
ENDING: {ending}
'''

def load_data(path: int):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

def run(instance, client):
    prompt = template(instance['precontext'],
                instance['sentence'],
                instance['ending'])
    response = client.responses.create(
                                        model="Qwen/Qwen3-0.6B",
                                        instructions=SYSTEM_PROMPT,
                                        input=prompt)
    print("Completion result:", response)
    return response.output_text

if __name__ == "__main__":
    dp = "sample_data.json"
    data = load_data(dp)
    for instance in tqdm(data.values()):
        result = run(instance, client)
        print(result.split("SCORE: ")[-1], instance["average"])