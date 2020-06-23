from datetime import timedelta
from threading import Lock

import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc

import model

# The global parameter server instance.
param_network = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()


class MasterNetwork:
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.model = model.MotionModel(input_dim, hidden_dim, layer_dim, output_dim)

    def forward(self, input):
        return self.model.forward(input)

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        return dist_autograd.get_gradients(cid)

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes paramters remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs


def get_parameter_network(input_dim, hidden_dim, layer_dim, output_dim):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_network
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_network:
            # construct it once
            param_network = MasterNetwork(input_dim, hidden_dim, layer_dim, output_dim)
        return param_network


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print('PS master initializing RPC')
    rpc.init_rpc(name='parameter_server', rank=rank, world_size=world_size)
    rpc._set_rpc_timeout(timedelta(seconds=60))
    print('RPC initialized! Running parameter server...')
    rpc.shutdown(graceful=True)
    print('RPC shutdown on parameter server.')
