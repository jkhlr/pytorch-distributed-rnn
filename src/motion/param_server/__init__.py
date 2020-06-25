import os

import torch
import torch.multiprocessing as mp

import dataset
from param_server.master import run_parameter_server
from param_server.worker import run_worker


def add_sub_command(parent_parser):
    parser = parent_parser.add_parser('parameter-server')
    parser.add_argument(
        '--world-size',
        type=int,
        required=True,
        help='''Total number of participating processes. Should be the sum of
        master node and all training nodes.''')
    parser.add_argument(
        '--rank',
        type=int,
        required=True,
        help='Global rank of this process. Pass in 0 for master.')
    parser.add_argument(
        '--master-address',
        type=str,
        default='localhost',
        help='''Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.''')
    parser.add_argument(
        '--master-port',
        type=str,
        default='29500',
        help='''Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.''')

    parser.set_defaults(func=execute)


def execute(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port

    processes = []
    world_size = args.world_size
    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        torch.set_num_threads(args.num_threads)
        training_set, validation_set, test_set = dataset.MotionDataset.load(args.dataset_path, output_path=args.output_path)
        # start training worker on this node
        p = mp.Process(
            target=run_worker,
            args=(
                args.rank,
                world_size,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                training_set.num_features,
                args.hidden_units,
                args.stacked_layer,
                len(dataset.MotionDataset.LABELS),
                training_set,
                validation_set,
                test_set))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
