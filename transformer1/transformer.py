import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from transformer1.encoder import Encoder
from transformer1.decoder import Decoder
from transformer1.mask import make_src_mask
from transformer1.mask import make_trg_mask

class Transformer(nn.Module):
    def __init__(self,
                 enc_input_dim, dec_output_dim,
                 enc_hid_dim,   dec_hid_dim,
                 enc_n_layer,   dec_n_layer,
                 enc_n_head,    dec_n_head,
                 enc_pf_dim,    dec_pf_dim,
                 enc_dropout,   dec_dropout,
                 enc_max_length, dec_max_length,
                 src_pad_idx,    trg_pad_idx
                 ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_dim=enc_input_dim,
                               hid_dim=enc_hid_dim,
                               n_layer=enc_n_layer,
                               n_head=enc_n_head,
                               pf_dim=enc_pf_dim,
                               dropout=enc_dropout,
                               max_length=enc_max_length)
        self.decoder = Decoder(output_dim=dec_output_dim,
                               hid_dim=dec_hid_dim,
                               n_layer=dec_n_layer,
                               n_head=dec_n_head,
                               pf_dim=dec_pf_dim,
                               dropout=dec_dropout,
                               max_length=dec_max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        src_mask = make_src_mask(src, self.src_pad_idx)
        trg_mask = make_trg_mask(trg, self.trg_pad_idx)

        # enc_src: [batch_size, src_len, hid_dim]
        enc_src = self.encoder(src, src_mask)

        # output:    [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_head, trg_len, trg_len]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
