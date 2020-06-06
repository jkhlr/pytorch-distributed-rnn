import logging
from itertools import product

import torch
from torch.utils import data
from torch.utils.data.dataset import random_split

types = list(product(("X", "y"), ("train", "test")))


class MotionDataset(data.Dataset):
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.seq_length = self.X_train.shape[1]
        self.num_features = self.X_train.shape[2]

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return len(self.X_train)

    def test_data(self):
        return self.X_test, self.y_test

    def random_split(self, validation_fraction=0.05):
        validation_examples = int(len(self) * validation_fraction)
        training_examples = len(self) - validation_examples
        return random_split(self, [training_examples, validation_examples])

    @classmethod
    def load(cls, base_path, output_path):
        if cls.processed_data_exists(base_path):
            logging.info("Preprocessed data found. Skip preprocessing.")
            input_data = {f"{data_type}_{set_type}": torch.load(base_path / f"{data_type}_{set_type}.pt") for
                          (data_type, set_type) in types}
            return cls(input_data["X_train"], input_data["y_train"], input_data["X_test"], input_data["y_test"])

        if output_path is None:
            output_path = base_path
        # only import if necessary
        logging.info("No processed data found. Preprocess raw data...")
        from processor import MotionDataProcessor
        processor = MotionDataProcessor()
        (X_train, y_train), (X_test, y_test) = processor.process_data(base_path)
        torch.save(X_train, output_path / "X_train.pt")
        torch.save(y_train, output_path / "y_train.pt")
        torch.save(X_test, output_path / "X_test.pt")
        torch.save(y_test, output_path / "y_test.pt")
        return cls(X_train, y_train, X_test, y_test)

    @classmethod
    def processed_data_exists(cls, base_path):
        paths = [base_path / f"{data_type}_{set_type}.pt" for (data_type, set_type) in types]
        return all(map(lambda path: path.exists(), paths))
