import os
import sys
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer.attention import MultiHeadAttention
from transformer.feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(num_head=n_head, hidden_size=d_model, dropout=dropout)
        self.pos_ffn  = PositionwiseFeedForward(d_input=d_model, d_hidden=d_inner, dropout=dropout)

    def forward(self, enc_input, enc_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(q=enc_input, k=enc_input, v=enc_input, mask=enc_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
