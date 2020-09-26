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

from cov2seq.encoder import Encoder
from cov2seq.decoder import Decoder


class Seq2seq(nn.Module):
    def __init__(self,
                 enc_embeddings,  dec_embeddings,
                 enc_input_dim,   dec_output_dim,
                 enc_emb_dim,     dec_emb_dim,
                 enc_hid_dim,     dec_hid_dim,
                 enc_n_layer,     dec_n_layer,
                 enc_kernel_size, dec_kernel_size,
                 enc_dropout,     dec_dropout,
                 dec_trg_pad_idx,
                 enc_max_length,  dec_max_length
                 ):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(enc_embeddings, enc_input_dim,  enc_emb_dim, enc_hid_dim, enc_n_layer, enc_kernel_size, enc_dropout, enc_max_length)
        self.decoder = Decoder(dec_embeddings, dec_output_dim, dec_emb_dim, dec_hid_dim, dec_n_layer, dec_kernel_size, dec_dropout, dec_trg_pad_idx, dec_max_length)

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len-1]  #  (<eos> token sliced off the end)

        # encoder_conved:   [batch_size, src_len, emb_dim]
        # encoder_combined: [batch_size, src_len, emb_dim]
        encoder_conved, encoder_combined = self.encoder(src)

        # output:    [batch_size, trg_len-1, output_dim]
        # attention: [batch_size, trg_len-1, src_len]
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        return output, attention
