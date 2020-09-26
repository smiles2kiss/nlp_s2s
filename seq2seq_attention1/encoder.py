import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embeddings, input_size, embedding_size, hidden_size, n_layer, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_dim = input_size
        self.embedding_size = embedding_size
        self.n_layer = n_layer
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(input_dim, emb_dim)
        # self.embedding = nn.Embedding(input_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        # self.embedding.requires_grad = False

        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=n_layer,
                          dropout=dropout,
                          bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    # Attention
    def forward(self, input_ids, input_len, hidden=None):
        # input_ids: [seq_len, batch_size]
        # emb:       [seq_len, batch_size, hidden_size]
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        # rnn_output: [seq_len, batch_size, n_direction*hidden_size]
        # hidden:     [n_layer*n_direction, batch_size, hidden_size]
        rnn_input = nn.utils.rnn.pack_padded_sequence(emb, input_len, enforce_sorted=False)
        rnn_output, hidden = self.gru(rnn_input, hidden)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)

        # hidden: [batch_size, 2*hidden_size]
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden: [batch_size, hidden_size]
        hidden = self.linear(hidden)
        hidden = torch.tanh(hidden)
        return rnn_output, hidden
