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

from transformer.attention import MultiHeadAttention
from transformer.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_q, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(num_head=n_head, hidden_size=d_model, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_input=d_model, d_hidden=d_inner, dropout=dropout)

    def forward(self, enc_input, enc_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(q=enc_input, k=enc_input, v=enc_input, mask=enc_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_q, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(num_head=n_head, hidden_size=d_model, d_q=d_q, d_k=d_k, d_v=d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(num_head=n_head, hidden_size=d_model, d_q=d_q, d_k=d_k, d_v=d_v, dropout=dropout)
        self.pos_ffn  = PositionwiseFeedForward(d_input=d_model, d_hidden=d_inner)

    def forward(self, dec_input, enc_output, dec_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(q=dec_input,  k=dec_input,  v=dec_input,  mask=dec_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(q=dec_output, k=enc_output, v=enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(enc_output)
        return dec_output, dec_slf_attn, dec_enc_attn
