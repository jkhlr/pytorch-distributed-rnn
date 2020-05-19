from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CoronaVirusPredictor(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, seq_len: int, n_layers: int = 2):
        super(CoronaVirusPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, dropout=0.5)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
        self.hidden = self.__initial_hidden_state()

    def reset_hidden_state(self):
        self.hidden = self.__initial_hidden_state()

    def forward(self, sequences: torch.FloatTensor):
        lstm_out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1), self.hidden)
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

    def train_model(self, loader: DataLoader, epochs: int, checkpoint_dir=None) -> (nn.Module, np.ndarray):
        loss_fn = torch.nn.MSELoss(reduction='sum')
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        train_hist = np.zeros(epochs)
        for t in range(epochs):
            for train_data, train_labels in loader:
                self.reset_hidden_state()
                optimiser.zero_grad()

                y_pred = self(train_data)
                loss = loss_fn(y_pred.float(), train_labels)
                train_hist[t] = loss.item()
                loss.backward()

                optimiser.step()

            print(f'Epoch {t} train loss: {loss.item()}')
            if checkpoint_dir and (t % 10 == 9 or t == epochs - 1):
                checkpoint = {
                    "epoch": t,
                    "model_state": self.state_dict(),
                    "optimizer_state": optimiser.state_dict(),
                    "loss": loss
                }
                torch.save(
                    checkpoint,
                    Path(checkpoint_dir)/f'checkpoint-epoch-{t}.pt'
                )

        return self.eval(), train_hist

    def __initial_hidden_state(self) -> (torch.FloatTensor, torch.FloatTensor):
        return (torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
                torch.zeros(self.n_layers, self.seq_len, self.n_hidden))
