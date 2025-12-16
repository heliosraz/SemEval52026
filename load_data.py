import pandas as pd
#from pathlib import Path
#from datasets import Dataset, DatasetDict
import json

def load_data(path):
    with open(path) as f:
        big_json = json.load(f)
        data = {id:{'precontext':big_json[id]['precontext'], 'sentence':big_json[id]['sentence'], 
                    'ending':big_json[id]['ending'],'average':big_json[id]['average'],
                    'stdev':big_json[id]['stdev'], 'example_sentence':big_json['example_sentence']
                    } for id in big_json}
    return data

def make_dataset(path):
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    new_data = data[['context', 'example_sentence', 'average', 'stdev', 'homonym']]
    return new_data
