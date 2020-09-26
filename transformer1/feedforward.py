import torch
import torch.nn as nn

class PositionwisedFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super(PositionwisedFeedforwardLayer, self).__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len, hid_dim]

        # x: [batch_size, seq_len, pf_dim]
        x = self.fc_1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # x: [batch_size, seq_len, hid_dim]
        x = self.fc_2(x)
        return x
