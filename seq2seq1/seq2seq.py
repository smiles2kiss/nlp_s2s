import os
import sys
import random
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from seq2seq1.encoder import Encoder
from seq2seq1.decoder import Decoder


class Seq2seq(nn.Module):
    # Sequence to Sequence Learning with Neural Networks
    def __init__(self,
                 enc_embeddings, dec_embeddings,
                 input_dim,   output_dim,
                 enc_emb_dim, dec_emb_dim,
                 enc_hid_dim, dec_hid_dim,
                 enc_n_layer, dec_n_layer,
                 enc_dropout, dec_dropout,
                 enc_bidirectional):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(enc_embeddings, input_dim,  enc_emb_dim, enc_hid_dim, enc_n_layer, dropout=enc_dropout, bidirectional=enc_bidirectional)
        self.decoder = Decoder(dec_embeddings, output_dim, dec_emb_dim, dec_hid_dim, dec_n_layer, dropout=dec_dropout)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(1)
        seq_len    = tgt.size(0)
        max_len    = 32
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(seq_len, batch_size, tgt_vocab_size).cuda()
        hidden = self.encoder(src)
        context = hidden

        decoder_input = tgt[0, :]
        for t in range(1, seq_len):
            output, hidden = self.decoder(decoder_input, hidden, context)
            outputs[t] = output
            tearcher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = tgt[t] if tearcher_force else top1

        return outputs
