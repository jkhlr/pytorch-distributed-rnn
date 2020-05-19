#!/usr/bin/env python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def run(rank):
    # create local model
    model = ToyModel().to('cpu')
    print('rank ', rank, ' initial: ', sum(parameter.sum() for parameter in model.parameters()))
    # construct DDP model
    ddp_model = DDP(model)
    print('rank ', rank, ' initial: ', sum(parameter.sum() for parameter in ddp_model.parameters()))
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # forward pass
    inputs = torch.randn(20, 10).to('cpu')
    print('rank ', rank, ' inputs: ', inputs.sum())
    outputs = ddp_model(inputs)
    labels = torch.randn(20, 5).to('cpu')
    print('rank ', rank, ' labels: ', labels.sum())
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    print('rank ', rank, ' parameters: ', sum(parameter.sum() for parameter in ddp_model.parameters()))

def init_process(fn, backend='mpi'):
    dist.init_process_group(backend)
    fn(dist.get_rank())


if __name__ == "__main__":
    init_process(run)