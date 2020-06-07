import argparse
import json
import logging
from pathlib import Path

import torch

from dataset import MotionDataset
from model import MotionModel
from trainer import DDPTrainer, Trainer

SCRIPT_DIR = Path(__file__).absolute().parent
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'models'
DEFAULT_DATASET_PATH = SCRIPT_DIR / 'data'


def main():
    parser = argparse.ArgumentParser(description="SusML JKTM")
    parser.add_argument("--checkpoint-directory", default=DEFAULT_CHECKPOINT_DIR, type=Path)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument("--output-path", default=None, type=Path)
    parser.add_argument("--stacked-layer", default=2, type=int)
    parser.add_argument("--hidden-units", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--validation-fraction", default=0.1, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.0025, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--log", default="INFO")
    parser.add_argument("--num-threads", default=4, type=int)
    parser.add_argument("--use-local", action='store_true')

    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    logging.getLogger().setLevel(args.log)

    dataset = MotionDataset.load(args.dataset_path, output_path=args.output_path, test=False)
    training_set, validation_set = dataset.random_split(validation_fraction=args.validation_fraction)

    test_set = MotionDataset.load(args.dataset_path, output_path=args.output_path, test=True)
    logging.info(f"Training set of size {len(training_set)}")
    logging.info(f"Validation set of size {len(validation_set)}")
    logging.info(f"Test set of size {len(test_set)}")

    logging.info("Create model")
    model = MotionModel(
        input_dim=dataset.num_features,
        hidden_dim=args.hidden_units,
        layer_dim=args.stacked_layer,
        output_dim=len(MotionDataset.LABELS),
    )

    if args.use_local:
        logging.info("Use local trainer")
        trainer = Trainer(
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_directory
        )
    else:
        logging.info("Use distributed trainer")
        trainer = DDPTrainer(
            model=model,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_directory
        )
    logging.info("Train model...")
    trained_model, train_history, validation_history = trainer.train(epochs=args.epochs)
    history = {"train_history": train_history, "validation_history": validation_history}
    with open('json_data.json', 'w') as file:
        json.dump(history, file)


if __name__ == '__main__':
    main()
