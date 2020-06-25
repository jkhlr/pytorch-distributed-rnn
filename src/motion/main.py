import argparse
import sys
from pathlib import Path

import param_server
import trainer

SCRIPT_DIR = Path(__file__).absolute().parent
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'models'
DEFAULT_DATASET_PATH = SCRIPT_DIR / 'data'


def main():
    parser = argparse.ArgumentParser(description='SusML JKTM')
    parser.add_argument('--checkpoint-directory', default=DEFAULT_CHECKPOINT_DIR, type=Path)
    parser.add_argument('--dataset-path', default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument('--output-path', default=None, type=Path)
    parser.add_argument('--stacked-layer', default=2, type=int)
    parser.add_argument('--hidden-units', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--validation-fraction', default=0.1, type=float)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--learning-rate', default=0.0025, type=float)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--log', default='INFO')
    parser.add_argument('--num-threads', default=4, type=int)

    sub_parser = parser.add_subparsers(title='Available commands', metavar='command [options ...]')

    param_server.add_sub_command(sub_parser)
    trainer.add_sub_commands(sub_parser)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
