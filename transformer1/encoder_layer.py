import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer1.attention import MultiHeadAttentionLayer
from transformer1.feedforward import PositionwisedFeedforwardLayer

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_head,
                 pf_dim,
                 dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim=hid_dim, n_head=n_head, dropout=dropout)
        self.positionwise_feedforward = PositionwisedFeedforwardLayer(hid_dim=hid_dim, pf_dim=pf_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src:      [batch_size, src_len, hid_dim]
        # src_mask: [batch_size, src_len]

        # _src: [batch_size, src_len, hid_dim]
        _src, _ = self.self_attention(query=src, key=src, value=src, mask=src_mask)

        # src: [batch_size, src_len, hid_dim]
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # _src: [batch_size, src_len, hid_dim]
        _src = self.positionwise_feedforward(src)

        # src: [batch_size, src_len, hid_dim]
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src
