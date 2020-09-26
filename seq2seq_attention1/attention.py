import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(2 * enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.fc   = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden:          [batch_size, dec_hid_dim]
        # encoder_outputs: [seq_len, batch_size, 2 * enc_hid_dim]
        seq_len = encoder_outputs.size(0)

        # repeat decoder hidden state seq_len times
        # hidden:          [batch_size, seq_len, dec_hid_dim]
        # encoder_outputs: [batch_size, seq_len, 2 * enc_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy: [batch_size, seq_len, 2 * enc_hid_dim + dec_hid_dim]
        # energy: [batch_size, seq_len, dec_hid_dim]
        energy = torch.cat((hidden, encoder_outputs), dim=2)
        energy = self.attn(energy)
        energy = torch.tanh(energy)

        # attention: [batch_size, seq_len, 1]
        # attention: [batch_size, seq_len]
        attention = self.fc(energy)
        attention = attention.squeeze(2)
        attention = F.softmax(attention)
        return attention
