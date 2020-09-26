import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embeddings, input_dim, emb_dim, hid_dim, n_layer, dropout=0.1):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.n_direction = 2

        # self.embedding = nn.Embedding(input_dim, emb_dim)
        # self.embedding = nn.Embedding(input_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        # self.embedding.requires_grad = False

        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=hid_dim,
                          num_layers=n_layer,
                          dropout=dropout,
                          bidirectional=True)
        self.linear = nn.Linear(2 * hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        # input_ids: [seq_len, batch_size]
        # emb:       [seq_len, batch_size, hidden_size]
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        # rnn_output: [seq_len, batch_size, n_direction*hidden_size]
        # hidden:     [n_direction, batch_size, hidden_size]
        rnn_output, hidden = self.rnn(emb)

        # hidden:     [batch_size, hidden_size]
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.linear(hidden)
        hidden = F.tanh(hidden)
        return rnn_output, hidden
