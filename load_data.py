import pandas as pd
#from pathlib import Path
#from datasets import Dataset, DatasetDict
import json
from datasets import load_dataset

def load_data(path):
    with open(path) as f:
        big_json = json.load(f)
        data = {id:{'precontext':big_json[id]['precontext'], 
                    'sentence':big_json[id]['sentence'], 
                    'ending':big_json[id]['ending'],
                    'average':big_json[id]['average'],
                    'stdev':big_json[id]['stdev'], 
                    'example_sentence':big_json['example_sentence']
                    } for id in big_json}
    return data

def make_dataset(path):
    data = load_dataset("json", data_dir = path)
    data['context'] = " ".join([data['precontext'],
                                data['sentence'],
                                data['ending']])
    return data
