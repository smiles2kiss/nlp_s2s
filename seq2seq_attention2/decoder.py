import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from seq2seq_attention2.attention import Dot_Attn
from seq2seq_attention2.attention import Concat_Attn
from seq2seq_attention2.attention import General_Attn


class Decoder(nn.Module):
    def __init__(self, embeddings, output_size, hidden_size, n_layer, dropout=0.1):
        super(Decoder, self).__init__()
        n_layer = 1
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layer     = n_layer
        self.is_attention = True
        dropout = 0 if n_layer == 1 else dropout
        print("decoder n_layer = ", n_layer)
        print("decoder with attention = ", self.is_attention)

        # self.embedding = nn.Embedding(output_dim, emb_dim)
        # self.embedding = nn.Embedding(output_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        self.emb_dropout = nn.Dropout(dropout)
        # self.embedding.requires_grad = False

        # Dot_Attn
        if self.is_attention:
            self.gru = nn.GRU(input_size=2*hidden_size,
                              hidden_size=hidden_size,
                              num_layers=n_layer,
                              dropout=dropout,
                              bidirectional=False)
        else:
            self.gru = nn.GRU(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=n_layer,
                              dropout=dropout,
                              bidirectional=False)

        # self.attn = Dot_Attn(hidden_size=hidden_size)
        # self.attn = Concat_Attn(hidden_size=hidden_size)
        self.attn = General_Attn(hidden_size=hidden_size)

        self.linear1 = nn.Linear(2*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size,   output_size)

    def forward(self, input_ids, decoder_hidden, encoder_outputs):
        # run decoder one step(word) at a time

        # decoder_hidden:  [n_layer, batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, hidden_size]

        # input_ids: [1, batch_size]
        # emb:       [1, batch_size, hidden_size]
        input_ids = input_ids.unsqueeze(0)
        emb = self.embedding(input_ids)
        emb = self.emb_dropout(emb)

        if self.is_attention:
            # context:      [1, batch_size, hidden_size]
            # concat_input: [1, batch_size, 2*hidden_size]
            context = self.attn(decoder_hidden, encoder_outputs)
            concat_input = torch.cat((emb, context), dim=2)

            # rnn_output:   [1, batch_size, hidden_size]
            # hidden:       [n_layer, batch_size, hidden_size]
            rnn_output, hidden = self.gru(concat_input, decoder_hidden)

            # concat_input: [batch_size, hidden_size]
            concat_output = rnn_output.squeeze(0)
        else:
            # rnn_output: [1,       batch_size, hidden_size]
            # hidden:     [n_layer, batch_size, hidden_size]
            rnn_output, hidden = self.gru(emb, decoder_hidden)

            # concat_output: [batch_size, 2*hidden_size]
            concat_output = torch.cat((rnn_output.squeeze(0), emb.squeeze(0)), dim=1)

            # concat_output: [batch_size, hidden_size]
            concat_output = self.linear1(concat_output)
            concat_output = torch.tanh(concat_output)

        # output: [batch_size, output_size]
        output = self.linear2(concat_output)
        return output, hidden
