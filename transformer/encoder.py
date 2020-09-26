import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer.layer import EncoderLayer
from transformer.encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self,
                 embeddings,
                 n_src_vocab, d_word_vec,
                 n_layer, n_head,
                 d_q, d_k, d_v,
                 d_model, d_inner,
                 pad_idx, dropout=0.1,
                 n_position=200,
                 max_seq_len=32):

        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # self.src_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx, _weight=embeddings.float())

        # self.pos_emb = PositionalEncoding(d_word_vec, n_position=n_position)
        self.pos_emb = nn.Embedding(max_seq_len, d_word_vec)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, d_inner=d_inner, n_head=n_head, d_q=d_q, d_k=d_k, d_v=d_v, dropout=dropout
                ) for _ in range(n_layer)
            ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_pad_mask, src_attn_mask, return_attn=False):
        enc_slf_attn_list = []

        batch_size = src_seq.size(0)
        src_len    = src_seq.size(1)

        # src_seq: [batch_size, src_len]
        # pos:     [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).cuda()

        # enc_output = self.src_emb(src_seq).cuda() + self.pos_emb(src_seq).cuda()
        emb_word = self.src_emb(src_seq)
        emb_posi = self.pos_emb(pos)

        enc_output = emb_word + emb_posi
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_outputs, enc_slf_attn = enc_layer(enc_input=enc_output, enc_attn_mask=src_attn_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attn else []

        if return_attn:
            return enc_output, enc_slf_attn_list
        return enc_output, enc_slf_attn_list

