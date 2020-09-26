import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer1.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layer,
                 n_head,
                 pf_dim,
                 dropout,
                 max_length=32):
        super(Encoder, self).__init__()

        self.tok_embedding = nn.Embedding(input_dim,  hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim=hid_dim,
                                                  n_head=n_head,
                                                  pf_dim=pf_dim,
                                                  dropout=dropout)
                                     for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

    def forward(self, src, src_mask):
        # src:      [batch_size, src_len, hid_dim]
        # src_mask: [batch_size, src_len]

        batch_size = src.size(0)
        src_len    = src.size(1)

        # ps: [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()

        # src: [batch_size, src_len, hid_dim]
        src = self.tok_embedding(src) * self.scale + self.pos_embedding(pos)
        src = self.dropout(src)

        # src: [batch_size, src_len, hid_dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        return src
