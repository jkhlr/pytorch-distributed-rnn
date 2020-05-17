#!/usr/bin/env python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank):
    # create local model
    model = nn.Linear(10, 10).to('cpu')
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # forward pass
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 10)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    print(list(model.parameters()))


if __name__ == "__main__":
    run(0)