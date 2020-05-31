import torch
import torch.nn as nn
import torch.nn.functional as F

class CoronaVirusPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(CoronaVirusPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        # out = F.relu(out)
        # out[:, -1, :] selects the last time step
        out = self.fc(out[:, -1, :])
        return out