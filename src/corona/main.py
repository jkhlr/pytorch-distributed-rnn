from pathlib import Path

from dataset import CoronaDataset
from model import CoronaVirusPredictor
from torchnet.dataset import ShuffleDataset
from trainer import DDPTrainer

SCRIPT_DIR = Path(__file__).absolute().parent
CHECKPOINT_DIR = SCRIPT_DIR / 'models'
DATASET_PATH = SCRIPT_DIR / 'data' / 'train.csv'


def main():
    dataset = CoronaDataset.load(DATASET_PATH)
    training_set, validation_set = dataset.random_split()

    model = CoronaVirusPredictor(
        n_features=dataset.num_features,
        seq_len=dataset.seq_length,
        n_hidden=1024,
        n_layers=1
    )

    trainer = DDPTrainer(
        model=model,
        training_set=ShuffleDataset(training_set),
        validation_set=validation_set,
        checkpoint_dir=CHECKPOINT_DIR
    )
    trained_model, history = trainer.train(epochs=50)


if __name__ == '__main__':
    main()
