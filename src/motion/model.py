import torch.nn as nn


class MotionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(MotionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Softmax is not required iff we use CrossEntropyLoss as the loss function during training

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        # out[:, -1, :] selects the last time step
        out = self.fc(out[:, -1, :])
        return out
