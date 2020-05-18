from torch.utils import data

from src.corona.dataset import CoronaDataset
from src.corona.model import CoronaVirusPredictor

MODEL_DIR = "./models"
DATA_DIR = "./data"

training_set = CoronaDataset.read_data(f"{DATA_DIR}/X_train.pt", f"{DATA_DIR}/y_train.pt")
training_generator = data.DataLoader(training_set)

model = CoronaVirusPredictor(n_features=training_set.num_features, n_hidden=512, seq_len=training_set.seq_length,n_layers=2)
trained_model, train_hist, test_hist = model.train_model(training_generator, MODEL_DIR, epochs=1)
