#!/usr/bin/env python
import horovod.torch as hvd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler


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


def run(rank, world_size):
    # create local model
    model = ToyModel()
    print(
        'rank ', rank,
        'initial:',
        sum(parameter.sum() for parameter in model.parameters())
    )
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    print(
        'rank', rank,
        'synced:',
        sum(parameter.sum() for parameter in model.parameters())
    )
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = hvd.DistributedOptimizer(
        optim.SGD(model.parameters(), lr=0.001),
        named_parameters=model.named_parameters()
    )

    dataset = ToyDataset(size=24)
    batch_size = 12 // world_size
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset, world_size, rank)
    )

    for inputs, labels in loader:
        optimizer.zero_grad()
        print('rank', rank, 'inputs:', inputs.sum())
        outputs = model(inputs)
        print('rank', rank, 'labels:', labels.sum())
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
        print(
            'rank', rank,
            'parameters:',
            sum(parameter.sum() for parameter in model.parameters())
        )


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    hvd.init()
    run(hvd.rank(), hvd.size())
