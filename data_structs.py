import pandas as pd
from torch.utils.data import Dataset


class WordSenseData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "average": self.data.loc[idx, "average"],
            "stdev": self.data.loc[idx, "stdev"],
            "index": self.data.loc[idx, "index"],
            "homonym": self.data.loc[idx, "homonym"],
            "full_context": self.data.loc[idx, "full_context"],
            "judged_meaning": self.data.loc[idx, "judged_meaning"],
            "example_sentence": self.data.loc[idx, "example_sentence"],
        }


class AugWordSenseData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {col: self.data.loc[idx, col] for col in self.data.columns.tolist()}


class CrossAttentionData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "average": self.data.loc[idx, "average"],
            "stdev": self.data.loc[idx, "stdev"],
            "candidate": self.data.loc[idx, "candidate"],
            "full_context": self.data.loc[idx, "full_context"],
        }
