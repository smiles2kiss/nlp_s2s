import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embeddings, input_dim, emb_dim, hid_dim, n_layer, dropout=0.1, bidirectional=False):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layer = n_layer
        self.bidirectional = bidirectional

        # self.embedding = nn.Embedding(input_dim, emb_dim)
        # self.embedding = nn.Embedding(input_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        # self.embedding.requires_grad = False

        self.rnn = nn.LSTM(input_size=emb_dim,
                           hidden_size=hid_dim,
                           num_layers=n_layer,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(2*hid_dim, hid_dim)
        self.linear2 = nn.Linear(2*hid_dim, hid_dim)

    def forward(self, input_ids):
        # input_ids: [src_len, batch_size]
        # emb:       [src_len, batch_size, emb_dim]
        batch_size = input_ids.size(1)
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        if self.bidirectional:
            # rnn_out: [src_len, batch_size, hid_dim]
            # hidden:  [n_layer*n_direction, batch_size, hid_dim]
            # cell:    [n_layer*n_direction, batch_size, hid_dim]
            rnn_out, (hidden, cell) = self.rnn(emb)

            hidden = hidden.view(self.n_layer, 2, batch_size, self.hid_dim)
            cell   = cell.view(  self.n_layer, 2, batch_size, self.hid_dim)

            # rnn_out: [src_len, batch_size, hid_dim]
            # hidden:  [n_layer, batch_size, hid_dim]
            # cell:    [n_layer, batch_size, hid_dim]
            hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
            cell   = torch.cat((  cell[:, -2, :, :],   cell[:, -1, :, :]), dim=2)
            hidden = self.linear1(hidden)
            cell   = self.linear2(cell)
            hidden = F.tanh(hidden)
            cell   = F.tanh(cell)
        else:
            # rnn_out: [src_len, batch_size, hid_dim]
            # hidden:  [n_layer, batch_size, hid_dim]
            # cell:    [n_layer, batch_size, hid_dim]
            rnn_out, (hidden, cell) = self.rnn(emb)
        return hidden, cell
