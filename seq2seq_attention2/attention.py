import torch
import torch.nn as nn
import torch.nn.functional as F


class Dot_Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Dot_Attn, self).__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  [1,       batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]
        seq_len     = encoder_outputs.size(0)
        batch_size  = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        # dot product attention
        # hidden:           [batch_size, seq_len, hidden_size]
        # encoder_outputs:  [batch_size, seq_len, hidden_size]
        # attention_weight: [batch_size, seq_len, hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        decoder_hidden = decoder_hidden.transpose(0, 1).repeat(1, seq_len, 1)
        attention = encoder_outputs * decoder_hidden

        # attention_weight: [batch_size, seq_len]
        attention = torch.sum(attention, dim=2)
        attention = F.softmax(attention, dim=1)

        # attention_weight: [batch_size, seq_len, 1]
        attention = attention.unsqueeze(2)

        # context_vector: [batch_size, seq_len, hidden_size]
        context = attention * encoder_outputs

        # context: [batch_size, hidden_size]
        context = torch.sum(context, dim=1)
        # context: [1, batch_size, hidden_size]
        context = context.unsqueeze(0)
        return context


class Concat_Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Concat_Attn, self).__init__()
        self.attn = nn.Linear(2*hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  [1,       batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]
        seq_len = encoder_outputs.size(0)

        # repeat decoder hidden state seq_len times
        # hidden:          [batch_size, seq_len, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        decoder_hidden = decoder_hidden.repeat(seq_len, 1, 1).permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy: [batch_size, seq_len, 2*hidden_size]
        # energy: [batch_size, seq_len, hidden_size]
        energy = torch.cat((decoder_hidden, encoder_outputs), dim=2)
        energy = self.attn(energy)
        energy = torch.tanh(energy)

        # attention: [batch_size, seq_len, 1]
        # attention: [batch_size, seq_len]
        attention = self.fc(energy)
        attention = attention.squeeze(2)
        attention = F.softmax(attention, dim=1)

        # attn_weights:    [batch_size, 1, seq_len]
        attention = attention.unsqueeze(1)

        # context: [batch_size, 1, hidden_size]
        context = attention.bmm(encoder_outputs)

        # context: [1, batch_size, hidden_size]
        context = context.permute(1, 0, 2)
        return context


class General_Attn(nn.Module):
    def __init__(self, hidden_size):
        super(General_Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  [1,       batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]

        # decoder_hidden:  [batch_size, 1,       hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        # attention:       [batch_size, seq_len, hidden_size]
        decoder_hidden  = decoder_hidden.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attention = self.attn(encoder_outputs)

        # attention: [batch_size, seq_len]
        attention = torch.sum(decoder_hidden * attention, dim=2)
        attention = F.softmax(attention, dim=1)

        #  attention: [batch_size, 1, seq_len]
        attention = attention.unsqueeze(1)

        # context: [batch_size, 1, hidden_size]
        context = attention.bmm(encoder_outputs)

        # context: [1, batch_size, hidden_size]
        context = context.permute(1, 0, 2)
        return context
