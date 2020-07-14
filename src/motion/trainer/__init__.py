import json
import logging

from dataset import MotionDataset
from model import MotionModel
from trainer.base import Trainer
from trainer.ddp import DDPTrainer
from trainer.horovod import HorovodTrainer

def add_sub_commands(sub_parser):
    create_parser(sub_parser, 'local', Trainer)
    create_parser(sub_parser, 'distributed', DDPTrainer)
    create_parser(sub_parser, 'horovod', HorovodTrainer)


def create_parser(sub_parser, name, trainer_class):
    parser = sub_parser.add_parser(name)
    parser.set_defaults(func=lambda args: train(args, trainer_class))


def train(args, trainer):
    logging.getLogger().setLevel(args.log)

    training_set, validation_set, test_set = MotionDataset.load(
        args.dataset_path,
        output_path=args.output_path
    )

    logging.info(f'Training set of size {len(training_set)}')
    if args.no_validation:
        validation_set = None
        test_set = None
    else:
        logging.info(f'Validation set of size {len(validation_set)}')
        logging.info(f'Test set of size {len(test_set)}')

    model = MotionModel(
        input_dim=training_set.num_features,
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
    trainer = trainer(**trainer_args)

    logging.info(f'Training model for {args.epochs} epochs...')
    _, train_history, validation_history = trainer.train(epochs=args.epochs)
    history = {'train_history': train_history,
               'validation_history': validation_history}
    with open('history.json', 'w') as file:
        json.dump(history, file)
