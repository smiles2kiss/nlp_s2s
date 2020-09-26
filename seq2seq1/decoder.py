import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, embeddings, output_dim, emb_dim, hid_dim, n_layer, dropout=0.1):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim

        # self.embedding = nn.Embedding(output_dim, emb_dim)
        # self.embedding = nn.Embedding(output_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        # self.embedding.requires_grad = False

        self.rnn = nn.GRU(input_size=emb_dim + hid_dim,
                          hidden_size=hid_dim,
                          num_layers=n_layer,
                          dropout=dropout,
                          bidirectional=False)
        self.linear = nn.Linear(emb_dim + 2*hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, hidden, context):
        # input_ids: [batch_size]
        # hidden:    [n_layer, batch_size, hid_dim]
        # context:   [n_layer, batch_size, hid_dim]

        # input_ids: [1, batch_size]
        input_ids = input_ids.unsqueeze(0)

        # emb: [1, batch_size, emb_dim]
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        # emb: [1, batch_size, emb_dim + hid_dim]
        emb_con = torch.cat((emb, context[-1].unsqueeze(0)), dim=2)

        # rnn_out: [n_layer, batch_size, hid_dim]
        # hidden:  [n_layer, batch_size, hid_dim]
        rnn_out, hidden = self.rnn(emb_con.contiguous(), hidden.contiguous())

        # output: [batch_size, emb_dim + hid_dim + hid_dim]
        output = torch.cat((emb.squeeze(0), hidden[-1], context[-1]), dim=1)

        # pred: [batch_size, output_dim]
        pred = self.linear(output)
        return pred, hidden

