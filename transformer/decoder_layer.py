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


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(num_head=n_head, hidden_size=d_model, dropout=dropout)
        self.enc_attn = MultiHeadAttention(num_head=n_head, hidden_size=d_model, dropout=dropout)
        self.pos_ffn  = PositionwiseFeedForward(d_input=d_model, d_hidden=d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, dec_attn_mask=None, dec_enc_attn_mask=None):
        # dec_input: [batch_size, trg_len]
        # dec_attn_mask: [batch_size, trg_len, trg_len]
        dec_output, dec_slf_attn = self.slf_attn(q=dec_input,  k=dec_input,  v=dec_input,  mask=dec_attn_mask)
        # dec_output:   [batch_size, trg_len, hidden_size]
        # dec_slf_attn: [batch_size, num_head, trg_len, trg_len]

        # dec_output: [batch_size, trg_len, hidden_size]
        # enc_output: [batch_size, src_len, hidden_size]
        # dec_enc_attn_mask: [batch_size, src_len, trg_len]
        dec_output, dec_enc_attn = self.enc_attn(q=dec_output, k=enc_output, v=enc_output, mask=dec_enc_attn_mask)
        # dec_output:   [batch_size, trg_len, hidden_size]
        # dec_enc_attn: [batch_size, num_head, trg_len, src_len]

        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
