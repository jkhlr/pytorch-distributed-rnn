from os.path import dirname, realpath
from pathlib import Path
from torchnet.dataset import ShuffleDataset

from dataset import CoronaDataset
from model import CoronaVirusPredictor
from trainer import DDPTrainer

SCRIPT_DIR = Path(__file__).absolute().parent
CHECKPOINT_DIR = SCRIPT_DIR / 'models'
DATASET_PATH = SCRIPT_DIR / 'data' / 'train.csv'

training_set = CoronaDataset.load(DATASET_PATH)
model = CoronaVirusPredictor(
    n_features=training_set.num_features,
    seq_len=training_set.seq_length,
    n_hidden=128,
    n_layers=3
)

trainer = DDPTrainer(
    model=model,
    training_set=ShuffleDataset(training_set),
    checkpoint_dir=CHECKPOINT_DIR
)
trained_model, history = trainer.train(epochs=1)
