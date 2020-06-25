import logging

import torch
from torch.utils import data
from torch.utils.data.dataset import random_split

from processor import MotionDataProcessor


class MotionDataset(data.Dataset):
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.seq_length = self.features.shape[1]
        self.num_features = self.features.shape[2]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

    def random_split(self, validation_fraction):
        validation_examples = int(len(self) * validation_fraction)
        training_examples = len(self) - validation_examples
        return random_split(self, [training_examples, validation_examples])

    @classmethod
    def load(cls, base_path, output_path=None, validation_fraction=0.05):
        types = ["train", "validation", "test"]
        paths = [cls.get_data_path(base_path, type) for type in types]
        datasets = []
        for feature_path, label_path in paths:
            if cls.processed_data_exists([feature_path, label_path]):
                features = torch.load(feature_path)
                labels = torch.load(label_path)
                datasets.append(cls(features, labels))

        # return only if all three were found
        if len(datasets) == 3:
            logging.info("Preprocessed data found. Skip preprocessing.")
            return datasets

        if output_path is None:
            output_path = base_path

        logging.info("No processed data found. Preprocess raw data...")
        processor = MotionDataProcessor()
        (X_train, y_train), (X_validation, y_validation), (X_test, y_test) = processor.process_data(base_path, validation_fraction)
        torch.save(X_train, output_path / "X_train.pt")
        torch.save(y_train, output_path / "y_train.pt")
        torch.save(X_validation, output_path / "X_validation.pt")
        torch.save(y_validation, output_path / "y_validation.pt")
        torch.save(X_test, output_path / "X_test.pt")
        torch.save(y_test, output_path / "y_test.pt")
        return cls(X_train, y_train), cls(X_validation, y_validation), cls(X_test, y_test)

    @classmethod
    def processed_data_exists(cls, paths):
        return all(map(lambda path: path.exists(), paths))

    @classmethod
    def get_data_path(cls, base_path, data_type):
        return base_path / f"X_{data_type}.pt", base_path / f"y_{data_type}.pt"
