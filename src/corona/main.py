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
    parser.add_argument("--checkpoint-directory", default=DEFAULT_CHECKPOINT_DIR, type=Path)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, type=Path)
    parser.add_argument("--stacked-layer", default=3, type=int)
    parser.add_argument("--hidden-units", default=256, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--validation-fraction", default=0.05, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=1e-6, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)

    args = parser.parse_args()

    print("Start DataLoader")
    dataset = CoronaDataset.load(args.dataset_path)
    #training_set, validation_set = dataset.random_split(validation_fraction=args.validation_fraction)

    print("Create model")
    model = CoronaVirusPredictor(
        n_features=dataset.num_features,
        seq_len=dataset.seq_length,
        n_hidden=args.hidden_units,
        dropout=args.dropout,
        n_layers=args.stacked_layer
    )

    print("Create trainer")
    trainer = DDPTrainer(
        model=model,
        training_set=ShuffleDataset(dataset),
        validation_set=None,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_directory
    )
    print("Train model...")
    trained_model, history, validation_history = trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()
