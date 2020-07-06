#!/usr/bin/env python
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
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
    torch.manual_seed(0)
    np.random.seed(0)
    model = ToyModel()
    print(
        'rank', rank,
        'initial:',
        sum(parameter.sum() for parameter in model.parameters())
    )
    # construct DDP model
    ddp_model = DDP(model)
    print(
        'rank', rank,
        'synced:',
        sum(parameter.sum() for parameter in ddp_model.parameters())
    )
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()

    dataset = ToyDataset(size=24)
    batch_size = 12 // world_size
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        #sampler=DistributedSampler(dataset, world_size, rank, shuffle=False)
    )

    for inputs, labels in loader:
        optimizer.zero_grad()
        print(
            'rank', rank,
            'grad:', optimizer.param_groups[0]['params'][0].grad.sum()
            if optimizer.param_groups[0]['params'][0].grad is not None else None
        )
        # print('rank', rank, 'inputs:', inputs.sum())
        outputs = ddp_model(inputs)
        # print('rank', rank, 'labels:', labels.sum())
        print('rank', rank, 'batch:', inputs.sum() + labels.sum())
        # backward pass
        loss = loss_fn(outputs, labels)
        print('rank', rank, 'loss:', loss.item())
        loss.backward()
        # update parameters

        optimizer.step()
        print(
            'rank', rank,
            'parameters:',
            sum(parameter.sum() for parameter in ddp_model.parameters())
        )
        print(
            'rank', rank,
            'grad:', optimizer.param_groups[0]['params'][0].grad.sum()
            if optimizer.param_groups[0]['params'][0].grad is not None else None
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    dist.init_process_group('mpi')
    run(dist.get_rank(), dist.get_world_size())
