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

from seq2seq_attention.encoder import Encoder
from seq2seq_attention.decoder import Decoder


class Seq2seq(nn.Module):
    """ Neural Machine Translation by Jointly Learning to Align and Translate """
    def __init__(self,
                 enc_embeddings, dec_embeddings,
                 input_dim,   output_dim,
                 enc_emb_dim, dec_emb_dim,
                 enc_hid_dim, dec_hid_dim,
                 attn_dim,
                 enc_n_layer, dec_n_layer,
                 enc_dropout, dec_dropout):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(enc_embeddings, input_dim,  enc_emb_dim, enc_hid_dim, enc_n_layer, enc_dropout)
        self.decoder = Decoder(dec_embeddings, output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, attn_dim, dec_n_layer, dec_dropout)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        seq_len    = tgt.size(0)
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(seq_len, batch_size, tgt_vocab_size).cuda()
        encoder_outputs, hidden = self.encoder(src)

        decoder_input = tgt[0, :]
        for t in range(1, seq_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.max(dim=1)[1]
            decoder_input = tgt[t] if teacher_force else top1
        return outputs
