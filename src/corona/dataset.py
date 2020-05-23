import torch
from torch.utils import data
from torch.utils.data.dataset import random_split

from processor import CoronaDataProcessor


class CoronaDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.seq_length = self.X.shape[1]
        self.num_features = self.X.shape[2]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def random_split(self, fraction=0.8):
        training_examples = int(len(self) * fraction)
        validation_examples = int(len(self) * (1 - fraction))
        return random_split(self, [training_examples, validation_examples])

    @classmethod
    def load(cls, csv_path):
        feature_path, label_path = cls.get_processed_data_path(csv_path)
        if cls.processed_data_exists(csv_path):
            return cls(X=torch.load(feature_path), y=torch.load(label_path))

        processor = CoronaDataProcessor(window_size=7)
        X, y = processor.process_data(csv_path)
        torch.save(X, feature_path)
        torch.save(y, label_path)
        return cls(X=X, y=y)

    @classmethod
    def processed_data_exists(cls, csv_path):
        feature_path, label_path = cls.get_processed_data_path(csv_path)
        return feature_path.exists() and label_path.exists()

    @classmethod
    def get_processed_data_path(cls, csv_path):
        dataset_name = csv_path.stem
        data_dir = csv_path.absolute().parent
        return data_dir / f'X_{dataset_name}.pt', data_dir / f'y_{dataset_name}.pt'
