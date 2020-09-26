import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_dim    = attn_dim

        self.attn = nn.Linear(2 * self.enc_hid_dim + self.dec_hid_dim, self.attn_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  [batch_size, dec_hid_dim]
        # encoder_outputs: [seq_len, batch_size, 2 * enc_hid_dim]
        seq_len = encoder_outputs.size(0)

        # decoder_hidden: [batch_size, seq_len, dec_hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # encoder_outputs: [batch_size, seq_len, 2 * enc_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy: [batch_size, seq_len, 2 * enc_hid_dim + dec_hid_dim]
        energy = torch.cat((decoder_hidden, encoder_outputs), dim=2)

        # energy: [batch_size, seq_len, attn_dim]
        energy = self.attn(energy)
        energy = F.tanh(energy)

        # attention: [batch_size, seq_len]
        attention = torch.sum(energy, dim=2)
        attention = F.softmax(attention, dim=1)
        return attention

