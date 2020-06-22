import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import dataset
from param_server.server import run_parameter_server
from param_server.util import get_data_loader
from param_server.worker import run_worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--master-address",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master-port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")

    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port

    SCRIPT_DIR = Path(__file__).absolute().parent
    DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'models'
    DEFAULT_DATASET_PATH = SCRIPT_DIR / 'data'

    processes = []
    world_size = args.world_size
    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        train_set = dataset.MotionDataset.load(DEFAULT_DATASET_PATH, output_path=DEFAULT_DATASET_PATH, test=False)
        test_set = dataset.MotionDataset.load(DEFAULT_DATASET_PATH, output_path=DEFAULT_DATASET_PATH, test=True)

        # Get data to train on
        train_loader = get_data_loader(train_set, batch_size=256)
        test_loader = get_data_loader(test_set, batch_size=256)
        print(f'Batch size: {len(train_loader)}')
        # start training worker on this node
        p = mp.Process(
            target=run_worker,
            args=(
                args.rank,
                world_size,
                train_set.num_features,
                32,
                1,
                len(dataset.MotionDataset.LABELS),
                train_loader,
                test_loader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
