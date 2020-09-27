import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.dirname(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from transformer1.attention import MultiHeadAttentionLayer
from transformer1.feedforward import PositionwisedFeedforwardLayer

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_head,
                 pf_dim,
                 dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)

        self.self_attention    = MultiHeadAttentionLayer(hid_dim=hid_dim, n_head=n_head, dropout=dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim=hid_dim, n_head=n_head, dropout=dropout)
        self.positionwise_feedforward = PositionwisedFeedforwardLayer(hid_dim=hid_dim, pf_dim=pf_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg:      [batch_size, trg_len]
        # enc_src:  [batch_size, src_len, hid_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # _trg: [batch_size, trg_len, hid_dim]
        _trg, _ = self.self_attention(query=trg, key=trg, value=trg, mask=trg_mask)

        # trg: [batch_size, trg_len, hid_dim]
        trg = trg + self.dropout(_trg)
        trg = self.self_attn_layer_norm(trg)

        # _trg:      [batch_size, trg_len, hid_dim]
        # attention: [batch_size, n_head, trg_len, src_len]
        _trg, attention = self.encoder_attention(query=trg, key=enc_src, value=enc_src, mask=src_mask)

        # trg: [batch_size, trg_len, hid_dim]
        trg = trg + self.dropout(_trg)
        trg = self.enc_attn_layer_norm(trg)

        # _trg: [batch_size, trg_len, hid_dim]
        _trg = self.positionwise_feedforward(trg)

        # trg: [batch_size, trg_len, hid_dim]
        trg = trg + self.dropout(_trg)
        trg = self.ff_layer_norm(trg)

        # trg: [batch_size, trg_len, hid_dim]
        # attention: [batch_size, n_head, trg_len, src_len]
        return trg, attention
