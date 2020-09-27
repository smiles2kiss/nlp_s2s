import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer.decoder_layer import DecoderLayer
from transformer.encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self,
                 n_trg_vocab,
                 d_word_vec,
                 n_layer,
                 n_head,
                 d_model,
                 d_inner,
                 pad_idx,
                 dropout=0.1,
                 n_position=200,
                 max_seq_len=32):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        # self.pos_emb = PositionalEncoding(d_word_vec, n_position=n_position)
        self.pos_emb = nn.Embedding(max_seq_len, d_word_vec)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout
                ) for _ in range(n_layer)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_attn_mask, enc_output, dec_enc_attn_mask, return_attn=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        batch_size = trg_seq.size(0)
        trg_len    = trg_seq.size(1)
        trg_pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).cuda()

        # dec_output = self.tgt_emb(trg_seq).cuda() + self.pos_emb(trg_seq).cuda()
        emb_word = self.tgt_emb(trg_seq)
        emb_posi = self.pos_emb(trg_pos)

        dec_output = emb_word + emb_posi
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:

            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_input=dec_output, enc_output=enc_output, dec_attn_mask=trg_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attn else []
            dec_enc_attn_list += [dec_enc_attn] if return_attn else []

        if return_attn:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list

        return dec_output, dec_slf_attn_list
