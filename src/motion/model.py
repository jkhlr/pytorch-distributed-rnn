import torch
import torch.nn as nn

class MotionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(MotionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).requires_grad_()
        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        # out[:, -1, :] selects the last time step
        out = self.fc(out[:, -1, :])
        prediction = self.sigmoid(out)
        return prediction