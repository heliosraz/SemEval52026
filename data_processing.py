import json

def load_data(path: str):
    with open(path, "r") as fp:
        data = json.load(fp)
    return data

def load_system_prompt(id: str, path: str = "data/system_prompt.jsonl"):
    with open(path, "r") as fp:
        data = json.load(fp)

    for inst in data:
        if inst["id"]==id:
            return inst
    else:
        raise KeyError("Desired prompt id doesn't exist.")
    
def prompt_template(precontext, target, ending):
    return f'''
    PRE-CONTEXT: {precontext}
    TARGET SENTENCE: {target}
    ENDING: {ending}
'''
    