import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size, n_layer, dropout=0.1):
        super(Encoder, self).__init__()
        n_layer = 1
        self.hidden_size = hidden_size
        self.n_layer     = n_layer
        self.bidirectional = False
        dropout = 0 if n_layer == 1 else dropout
        print("encoder n_layer = ", n_layer)
        print("encoder self.bidirectional = ", self.bidirectional)

        # self.embedding = nn.Embedding(input_dim, emb_dim)
        # self.embedding = nn.Embedding(input_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        # self.embedding.requires_grad = False

        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=n_layer,
                          dropout=dropout,
                          bidirectional=self.bidirectional)
        self.linear = nn.Linear(2*hidden_size, hidden_size)

    # Dot_Attn
    def forward(self, input_ids, input_len, hidden=None):
        # input_ids: [seq_len, batch_size]
        # input_len: [batch_size]
        # emb:       [seq_len, batch_size, hidden_size]
        emb = self.embedding(input_ids)
        batch_size = input_ids.size(1)

        # rnn_output: [seq_len, batch_size, n_direction*hidden_size]
        # hidden:     [n_direction*n_layer, batch_size, hidden_size]
        packed = nn.utils.rnn.pack_padded_sequence(emb, input_len, enforce_sorted=False)
        rnn_output, hidden = self.gru(packed, hidden)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)

        if self.bidirectional:
            # sum bidirectional gpu outputs
            # rnn_output: [seq_len, batch_size, hidden_size]
            rnn_output = rnn_output[:, :, :self.hidden_size] + rnn_output[:, :, self.hidden_size:]

            # hidden: [n_layer, n_direction, batch_size, hidden_size]
            hidden = hidden.view(self.n_layer, -1, batch_size, self.hidden_size)

            # hidden: [n_layer, batch_size, 2*hidden_size]
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)

            # hidden: [n_layer, batch_size, hidden_size]
            hidden = self.linear(hidden)
            hidden = torch.tanh(hidden)
        return rnn_output, hidden
