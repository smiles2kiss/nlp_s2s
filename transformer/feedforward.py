import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_input, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_input, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_input)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        x = x + residual
        x = self.layer_norm(x)

        return x
