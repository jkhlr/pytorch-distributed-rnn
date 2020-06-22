from datetime import timedelta

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.distributed.optim import DistributedOptimizer

from param_server.server import get_parameter_server, ParameterServer
from param_server.util import remote_method


class Worker(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(input_dim, hidden_dim, layer_dim, output_dim)
        )

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output


def run_training_loop(rank, input_dim, hidden_dim, layer_dim, output_dim, train_loader, test_loader):
    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a server fashion.
    net = Worker(input_dim, hidden_dim, layer_dim, output_dim)
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
    loss_fn = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as cid:
            model_output = net(data)
            target = target.long().squeeze(1)
            loss = loss_fn(model_output, target)
            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}")
            dist_autograd.backward([loss])
            # Ensure that dist autograd ran successfully and gradients were
            # returned.
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}
            opt.step()

    print("Training complete!")
    print("Getting accuracy....")
    get_accuracy(test_loader, net)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


# Main loop for trainers.
def run_worker(rank, world_size, input_dim, hidden_dim, layer_dim, output_dim, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    rpc._set_rpc_timeout(timedelta(seconds=60))
    print(f"Worker {rank} done initializing RPC")

    run_training_loop(rank, input_dim, hidden_dim, layer_dim, output_dim, train_loader, test_loader)
    rpc.shutdown()
