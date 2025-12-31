import pandas as pd
#from pathlib import Path
#from datasets import Dataset, DatasetDict

def load_data(path):
    data = pd.read_json(path).transpose()
    data['context'] = data['precontext'] + ' ' + data['sentence'] + ' ' + data['ending']
    new_data = data[['context', 'example_sentence', 'average', 'stdev', 'homonym']]
    new_data = new_data.reset_index()
    return new_data
