import pandas as pd
#from pathlib import Path
#from datasets import Dataset, DatasetDict
import json
from datasets import load_dataset

def load_data(path):
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    new_data = data[['context', 'example_sentence', 'average', 'stdev', 'homonym']]
    return new_data
