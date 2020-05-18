from __future__ import annotations

import torch
from torch.utils import data


class CoronaDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.seq_length = self.X.shape[1]
        self.num_features = self.X.shape[2]

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.X)

    @classmethod
    def read_data(cls, feature_path: str, label_path: str) -> CoronaDataset:
        return cls(torch.load(feature_path), torch.load(label_path))
