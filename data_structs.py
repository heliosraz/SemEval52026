import pandas as pd
from torch.utils.data import Dataset


class WordSenseData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "average": self.data.iloc[idx]["average"],
            "stdev": self.data.iloc[idx]["stdev"],
            "index": self.data.iloc[idx]["index"],
            "homonym": self.data.iloc[idx]["homonym"],
            "full_context": self.data.iloc[idx]["full_context"],
            "judged_meaning": self.data.iloc[idx]["judged_meaning"],
            "example_sentence": self.data.iloc[idx]["example_sentence"],
        }


class AugWordSenseData(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {col: self.data.iloc[idx][col] for col in self.data.columns.tolist()}


class CrossAttentionData(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "average": self.data.iloc[idx]["average"],
            "stdev": self.data.iloc[idx]["stdev"],
            "candidate": self.data.iloc[idx]["candidate"],
            "full_context": self.data.iloc[idx]["full_context"],
        }
