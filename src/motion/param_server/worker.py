from datetime import timedelta

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DistributedSampler

from param_server.master import get_parameter_network, MasterNetwork
from param_server.util import remote_method
from trainer.base import Trainer
from trainer.formatter import TrainingMessageFormatter


class ParameterWorkerTrainer(Trainer):
    def __init__(self, rank, wold_size, model, training_set, batch_size, learning_rate, validation_set=None,
                 test_set=None,
                 checkpoint_dir=None):
        self.rank = rank
        self.world_size = wold_size
        super().__init__(
            model=model,
            training_set=training_set,
            validation_set=self._get_eval_set(validation_set),
            test_set=self._get_eval_set(test_set),
            batch_size=batch_size,
            learning_rate=learning_rate,
            checkpoint_dir=checkpoint_dir,
            sampler=DistributedSampler(training_set, num_replicas=self.world_size, rank=self.rank),
        )

    def _get_formatter(self, epochs):
        return TrainingMessageFormatter(epochs, self.rank)

    def _get_optimizer(self, model, lr):
        return DistributedOptimizer(optim.Adam, model.get_global_param_rrefs(), lr=lr)

    def _train_step(self, formatter):
        self.model.train()
        total_loss = 0.
        total_correct = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            with dist_autograd.context() as cid:
                output = self.model(data)
                target = target.long().squeeze(1)
                loss = self.loss_fn(output, target)
                total_loss += loss.item()
                correct = (torch.argmax(output, dim=1) == target).sum()
                total_correct += correct
                dist_autograd.backward([loss])
                # Ensure that dist autograd ran successfully and gradients were
                # returned.
                assert remote_method(
                    MasterNetwork.get_dist_gradients,
                    self.model.param_server_rref,
                    cid) != {}
                self.optimizer.step()
                print(formatter.train_progress_message(batch_idx=batch_idx, batches=len(self.train_loader),
                                                       training_examples=len(data), correct=correct,
                                                       loss=loss.item()))
        train_loss = total_loss / len(self.train_loader.dataset)
        train_acc = total_correct / len(self.train_loader.dataset)
        return train_loss, train_acc

    def _get_eval_set(self, eval_set):
        if self.rank is None or self.rank == 0:
            return eval_set
        else:
            return None

    def _save_checkpoint(self, epoch, loss, best=False):
        if self.rank != 0:
            return


class WorkerNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.param_server_rref = rpc.remote(
            'parameter_server', get_parameter_network, args=(input_dim, hidden_dim, layer_dim, output_dim)
        )

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            MasterNetwork.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def forward(self, x):
        model_output = remote_method(
            MasterNetwork.forward, self.param_server_rref, x)
        return model_output


def run_worker(rank, world_size, epochs, batch_size, learning_rate, input_dim, hidden_dim, layer_dim, output_dim,
               train_set, validation_set, test_set):
    print(f'Worker rank {rank} initializing RPC')
    rpc.init_rpc(
        name=f'trainer_{rank}',
        rank=rank,
        world_size=world_size)

    rpc._set_rpc_timeout(timedelta(seconds=60))
    print(f'Worker {rank} done initializing RPC')

    model = WorkerNetwork(input_dim, hidden_dim, layer_dim, output_dim)
    worker = ParameterWorkerTrainer(rank, world_size, model, train_set, batch_size, learning_rate, validation_set,
                                    test_set)
    worker.train(epochs)
    rpc.shutdown()
