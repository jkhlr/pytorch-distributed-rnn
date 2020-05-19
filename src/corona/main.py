from os.path import dirname, realpath
from pathlib import Path
from torch.utils import data

from processor import CoronaDataProcessor
from dataset import CoronaDataset
from model import CoronaVirusPredictor


SCRIPT_DIR = Path(dirname(realpath(__file__)))
DATA_DIR = SCRIPT_DIR / 'data'
MODEL_DIR = SCRIPT_DIR / 'models'

if not ((DATA_DIR/'X_train.pt').exists() and (DATA_DIR/'y_train.pt').exists()):
    processor = CoronaDataProcessor(window_size=7)
    processor.process_data(DATA_DIR)

training_set = CoronaDataset.read_data(DATA_DIR/'X_train.pt', DATA_DIR/'y_train.pt')
training_generator = data.DataLoader(training_set)

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir()

model = CoronaVirusPredictor(
    n_features=training_set.num_features,
    seq_len=training_set.seq_length,
    n_hidden=128,
    n_layers=3
)
model.train_model(training_generator, epochs=50, checkpoint_dir=MODEL_DIR)
