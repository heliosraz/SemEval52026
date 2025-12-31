import pandas as pd
#from pathlib import Path
#from datasets import Dataset, DatasetDict
import json
from datasets import load_dataset

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
        data = pd.DataFrame({'id': {'context': " ".join([data['precontext'],
                                data['sentence'],
                                data['ending']]),
                        'sentence':data['example_sentence'], 
                        'average':data[id]['average']
                        } for id in data})
    return data
