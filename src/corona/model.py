import torch
import torch.nn as nn

class CoronaVirusPredictor(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, seq_len: int, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, sequences):
        batch_size = sequences.size(0)
        h0, c0 = self._initial_hidden_state(batch_size)
        lstm_out, self.hidden = self.lstm(sequences, (h0, c0))
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred

    def _initial_hidden_state(self, batch_size) -> (torch.FloatTensor, torch.FloatTensor):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden),
                torch.zeros(self.n_layers, batch_size, self.n_hidden))
