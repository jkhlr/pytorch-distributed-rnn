import torch
import torch.nn as nn

class CoronaVirusPredictor(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, seq_len: int, n_layers):
        super().__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, dropout=0.1)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
        self.hidden = self.__initial_hidden_state()

    def reset_hidden_state(self):
        self.hidden = self.__initial_hidden_state()

    def forward(self, sequences: torch.FloatTensor):
        lstm_out, self.hidden = self.lstm(sequences.view(len(sequences), self.seq_len, -1), self.hidden)
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

    def __initial_hidden_state(self) -> (torch.FloatTensor, torch.FloatTensor):
        return (torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
                torch.zeros(self.n_layers, self.seq_len, self.n_hidden))
