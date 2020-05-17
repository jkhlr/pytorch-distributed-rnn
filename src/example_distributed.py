#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        for i in range(1, size):
            dist.send(tensor=tensor, dst=i)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(fn, backend='mpi'):
    dist.init_process_group(backend)
    fn(dist.get_rank(), dist.get_world_size())


if __name__ == "__main__":
    init_process(run)