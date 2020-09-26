import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer1.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layer,
                 n_head,
                 pf_dim,
                 dropout,
                 max_length=100):
        super(Decoder, self).__init__()

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim=hid_dim,
                                                  n_head=n_head,
                                                  pf_dim=pf_dim,
                                                  dropout=dropout)
                                     for _ in range(n_layer)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg:      [batch_size, trg_len]
        # enc_src:  [batch_size, src_len, hid_dim]
        # trg_mask: [batch_size, trg_len]
        # src_maks: [batch_size, src_mask]

        batch_size = trg.size(0)
        trg_len    = trg.size(1)

        # pos: [batch_size, trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).cuda()

        # trg: [batch_size, trg_len, hid_dim]
        trg = self.tok_embedding(trg) * self.scale + self.pos_embedding(pos)
        trg = self.dropout(trg)

        # trg: [batch_size, trg_len, hid_dim]
        # attention: [batch_size, n_head, trg_len, src_len]
        attention = None
        for layer in self.layers:
            trg, attention = layer(trg=trg, enc_src=enc_src, trg_mask=trg_mask, src_mask=src_mask)

       # output: [batch_size, trg_len, output_dim]
        output = self.fc_out(trg)

        return output, attention

