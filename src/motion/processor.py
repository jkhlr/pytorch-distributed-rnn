from pathlib import Path

import numpy as np
import torch


class MotionDataProcessor:
    """
    This class handles raw txt files provided by the UCI HAR Dataset.
    The result are torch float tensors.

    """
    TRAIN = "train/"
    TEST = "test/"

    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    def process_data(self, csv_path, validation_fraction=0.05):
        """
        Loads motion sensor data from multiple text data and returns torch float tensors.
        The processing is based on https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LSTM.ipynb

        :param csv_path: base path of the dataset
        :return: a training and test pair of features and labels
        """
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)

        X_train_signals_paths = [
            csv_path / self.TRAIN / "Inertial Signals/" / (signal + "train.txt") for signal in self.INPUT_SIGNAL_TYPES
        ]
        X_test_signals_paths = [
            csv_path / self.TEST / "Inertial Signals/" / (signal + "test.txt") for signal in self.INPUT_SIGNAL_TYPES
        ]

        X_train = self._load_X(X_train_signals_paths)
        X_test = self._load_X(X_test_signals_paths)

        y_train_path = csv_path / self.TRAIN / "y_train.txt"
        y_test_path = csv_path / self.TEST / "y_test.txt"

        y_train = self._load_y(y_train_path)
        y_test = self._load_y(y_test_path)

        train, valid = self._train_valid_split(X_train, y_train, validation_fraction)

        return train, valid, (X_test, y_test)

    def _load_X(self, X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

        return torch.FloatTensor(np.transpose(np.array(X_signals), (1, 2, 0)))

    def _load_y(self, y_path):
        file = open(y_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()

        # Substract 1 to each output class for friendly 0-based indexing
        return torch.LongTensor(y_ - 1)

    def _train_valid_split(self, features, labels, validation_fraction):
        num_examples = features.shape[0]
        assert num_examples == labels.shape[0], "Feature and labels must have equal size"
        indices = np.random.permutation(num_examples)
        validation_examples = int(num_examples * validation_fraction)
        valid_idx, train_idx = indices[:validation_examples], indices[validation_examples:]
        X_train, X_valid = features[train_idx, :], features[valid_idx, :]
        y_train, y_valid = labels[train_idx, :], labels[valid_idx, :]
        return (X_train, y_train), (X_valid, y_valid)
