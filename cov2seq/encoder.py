import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 embeddings,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layer,
                 kernel_size,
                 dropout,
                 max_length=100):
        super(Encoder, self).__init__()

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).cuda()
        # self.tok_embedding = nn.Embedding(input_dim,  emb_dim, _weight=embeddings.float())
        self.tok_embedding = nn.Embedding(input_dim,  emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hid_dim,
                      out_channels=2*hid_dim,
                      kernel_size=kernel_size,
                      padding=(kernel_size-1) // 2)
            for _ in range(n_layer)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        batch_size = src.size(0)
        src_len    = src.size(1)

        # create position tensor
        # pos: [batch_size, src_len]
        # pos: [0, 1, 2, ..., src_len-1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()

        # tok_emb: [batch_size, src_len, emb_dim]
        # pos_emb: [batch_size, src_len, emb_dim]
        tok_emb = self.tok_embedding(src)
        pos_emb = self.pos_embedding(pos)

        # emb: [batch_size, src_len, emb_dim]
        emb = tok_emb + pos_emb
        emb = self.dropout(emb)

        # conv_input: [batch_size, src_len, hid_dim]
        conv_input = self.emb2hid(emb)

        # conv_input: [batch_size, hid_dim, src_len]
        conv_input = conv_input.permute(0, 2, 1)
        conved = conv_input

        for i, conv in enumerate(self.convs):
            # conved: [batch_size, 2*hid_dim, src_len]
            conved = conv(self.dropout(conv_input))

            # conved: [batch_size, hid_dim, src_len)
            conved = F.glu(conved, dim=1)

            # conved: [batch_size, hid_dim, src_len]
            conved = (conved + conv_input) * self.scale

            # conv_input: [batch_size, hid_dim, src_len]
            conv_input = conved

        # conved: [batch_size, src_len, hid_dim]
        conved = conved.permute(0, 2, 1)

        # conved: [batch_size, src_len, emb_dim]
        conved = self.hid2emb(conved)

        # combined: [batch_size, src_len, emb_dim]
        combined = (conved + emb) * self.scale

        return conved, combined

