import argparse
from pathlib import Path

from dataset import CoronaDataset
from model import CoronaVirusPredictor
from torchnet.dataset import ShuffleDataset
from trainer import DDPTrainer

SCRIPT_DIR = Path(__file__).absolute().parent
DEFAULT_CHECKPOINT_DIR = SCRIPT_DIR / 'models'
DEFAULT_DATASET_PATH = SCRIPT_DIR / 'data' / 'train.csv'


def main():
    parser = argparse.ArgumentParser(description="SusML JKTM")
    parser.add_argument("--checkpoint-directory", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-directory", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--stacked-layer", default=3, type=int)
    parser.add_argument("--hidden-units", default=32, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--type", default="distributed", choices=["local", "distributed"])
    parser.add_argument("--validation-fraction", default=0.95, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=1e-6, type=float)

    args = parser.parse_args()

    dataset = CoronaDataset.load(DEFAULT_DATASET_PATH)
    training_set, validation_set = dataset.random_split(fraction=args.validation_fraction)

    model = CoronaVirusPredictor(
        n_features=dataset.num_features,
        seq_len=dataset.seq_length,
        n_hidden=args.hidden_units,
        n_layers=args.stacked_layer
    )

    trainer = DDPTrainer(
        model=model,
        training_set=ShuffleDataset(training_set),
        validation_set=validation_set,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_directory
    )
    trained_model, history = trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()
