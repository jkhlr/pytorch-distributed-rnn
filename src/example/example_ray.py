"""Example code for RaySGD Torch in the documentation.
It ignores yapf because yapf doesn't allow comments right after code blocks,
but we put comments right after code blocks to prevent large white spaces
in the documentation.
"""

# yapf: disable
# __torch_train_example__
import argparse

import numpy as np
import ray
import torch
import torch.nn as nn
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from torch.utils.data import Dataset, DataLoader


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class ToyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.features = torch.randn(size, 10)
        self.labels = torch.randn(size, 5)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.size


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, a, b, size=1000):
        x = np.arange(0, 10, 10 / size, dtype=np.float32)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(a * x + b)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


def model_creator(_):
    torch.manual_seed(0)
    np.random.seed(0)
    model = ToyModel()
    print(
        'initial:',
        sum(parameter.sum() for parameter in model.parameters())
    )
    return model
    # return nn.Linear(1, 1)


def optimizer_creator(model, _):
    return torch.optim.SGD(model.parameters(), lr=0.001)


def data_creator(_):
    batch_size = 12
    train_loader = DataLoader(
        shuffle=False,
        dataset=ToyDataset(size=24),
        # dataset=LinearDataset(2, 5, 1000),
        batch_size=batch_size
    )
    validation_loader = DataLoader(
        shuffle=False,
        dataset=ToyDataset(size=1),
        # dataset=LinearDataset(2, 5, 400),
        batch_size=batch_size
    )
    return train_loader, validation_loader


def run(rank, world_size):
    model = ToyModel()
    print(
        'rank', rank,
        'initial:',
        sum(parameter.sum() for parameter in model.parameters())
    )
    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    dataset = ToyDataset(size=24)
    batch_size = 12 // world_size
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # sampler=DistributedSampler(dataset, world_size, rank, shuffle=False)
    )

    for inputs, labels in loader:
        print('rank', rank, 'inputs:', inputs.sum())
        print('rank', rank, 'labels:', labels.sum())

    class operator(TrainingOperator):
        def train_batch(self, batch, batch_info):
            print(
                'grad:', optimizer.param_groups[0]['params'][0].grad.sum()
                if optimizer.param_groups[0]['params'][0].grad is not None else None
            )
            print(
                'batch:',
                sum(parameter.sum() for parameter in batch)
            )
            info = super().train_batch(batch, batch_info)
            print('loss:', info['train_loss'])
            print(
                'parameters:',
                sum(parameter.sum() for parameter in model.parameters())
            )
            print(
                'grad:', optimizer.param_groups[0]['params'][0].grad.sum()
                if optimizer.param_groups[0]['params'][0].grad is not None else None
            )
            return info

    trainer = TorchTrainer(
        model_creator=lambda _: model,
        data_creator=lambda _: (loader, None),
        optimizer_creator=lambda _, __: optimizer,
        loss_creator=nn.MSELoss,
        training_operator_cls=operator,
        num_workers=world_size,
        use_gpu=False,
        backend="gloo"
    )

    for i in range(1):
        stats = trainer.train()
        print(stats)
        print(
            'parameters:',
            sum(parameter.sum() for parameter in model.parameters())
        )

    trainer.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        required=False,
        type=str,
        help="the address to use for Ray")
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.")
    args, _ = parser.parse_known_args()

    torch.manual_seed(0)
    np.random.seed(0)
    ray.init(address=args.address)
    run(0, 1)
