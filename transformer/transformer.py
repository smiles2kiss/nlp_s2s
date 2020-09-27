import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.mask import get_pad_mask
from transformer.mask import get_subsequent_mask


class Transformer(nn.Module):
    def __init__(self,
                 n_src_vocab,
                 n_trg_vocab,
                 src_pad_idx,
                 trg_pad_idx,
                 d_word_vec=256,
                 d_model=256,
                 d_inner=512,
                 n_layer=3,
                 n_head=8,
                 dropout=0.1,
                 n_position=200):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_layer,
            n_head=n_head,
            pad_idx=src_pad_idx,
            dropout=dropout,
            n_position=n_position,
            max_seq_len=32
        )
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_layer,
            n_head=n_head,
            pad_idx=trg_pad_idx,
            n_position=n_position,
            dropout=dropout
        )

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        # for name, param in self.named_parameters():
        #     if param.dim() > 1:
        #         nn.init.xavier_normal(param)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src_seq, trg_seq):
        # src_seq: [batch_size, src_len]
        # trg_seq: [batch_size, trg_len]
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        # src_mask: [batch_size, 1,       src_len]
        # trg_mask: [batch_size, trg_len, trg_len]

        enc_output, *_ = self.encoder(src_seq=src_seq, src_attn_mask=src_mask)
        dec_output, *_ = self.decoder(trg_seq=trg_seq, trg_attn_mask=trg_mask, enc_output=enc_output, dec_enc_attn_mask=src_mask)
        # enc_output: [batch_size, src_len, hidden_size]
        # dec_output: [batch_size, trg_len, hidden_size]

        seq_logit = self.trg_word_prj(dec_output)
        # seq_logit: [batch_size, seq_len, trg_vocab_size]
        return seq_logit
