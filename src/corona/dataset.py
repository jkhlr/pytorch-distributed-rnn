import torch
from torch.utils import data
from torch.utils.data.dataset import random_split
import logging


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

    def random_split(self, validation_fraction=0.05):
        validation_examples = int(len(self) * validation_fraction)
        training_examples = len(self) - validation_examples
        return random_split(self, [training_examples, validation_examples])

    @classmethod
    def load(cls, csv_path):
        feature_path, label_path = cls.get_processed_data_path(csv_path)
        if cls.processed_data_exists(csv_path):
            logging.info("Preprocessed data found. Skip preprocessing.")
            return cls(X=torch.load(feature_path), y=torch.load(label_path))

        # only import if it is necessary
        # this allows us to use provide the data in environments where we cannot install sklearn and pandas
        logging.info("No processed data found. Preprocess raw data...")
        from processor import CoronaDataProcessor
        processor = CoronaDataProcessor(window_size=14)
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
