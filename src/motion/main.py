import argparse
import json
import logging
from pathlib import Path

from dataset import MotionDataset
from model import MotionModel
from trainer.base import Trainer
from trainer.ddp import DDPTrainer
from trainer.horovod import HorovodTrainer

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
    parser.add_argument('--trainer', default='local', type=str)
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log)

    dataset = MotionDataset.load(args.dataset_path, output_path=args.output_path, test=False)
    training_set, validation_set = dataset.random_split(validation_fraction=args.validation_fraction)
    test_set = MotionDataset.load(args.dataset_path, output_path=args.output_path, test=True)
    logging.info(f'Training set of size {len(training_set)}')
    logging.info(f'Validation set of size {len(validation_set)}')
    logging.info(f'Test set of size {len(test_set)}')

    model = MotionModel(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_units,
        layer_dim=args.stacked_layer,
        output_dim=len(MotionDataset.LABELS),
    )

    trainer_args = dict(
        model=model,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_directory
    )
    if args.trainer == 'local':
        logging.info('Use local trainer')
        trainer = Trainer(**trainer_args)
    elif args.trainer == 'ddp':
        logging.info('Use DDP trainer')
        trainer = DDPTrainer(**trainer_args)
    elif args.trainer == 'horovod':
        logging.info('Use Horovod trainer')
        trainer = HorovodTrainer(**trainer_args)
    else:
        raise ValueError(f'ERROR: Invalid trainer: {args.trainer}')

    logging.info(f'Training model for {args.epochs} epochs...')
    trained_model, train_history, validation_history = trainer.train(epochs=args.epochs)
    history = {'train_history': train_history, 'validation_history': validation_history}
    with open('history.json', 'w') as file:
        json.dump(history, file)


if __name__ == '__main__':
    main()
