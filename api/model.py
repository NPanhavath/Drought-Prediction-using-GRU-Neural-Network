import torch
import torch.nn as nn

class DroughtPredictorGRU(nn.Module):
    """GRU Time-Series Network for Drought Score Prediction."""
    def __init__(self, input_dim=18, hidden_size=64, output_dim=1):
        super(DroughtPredictorGRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_day_output = gru_out[:, -1, :]
        return self.linear(last_day_output)