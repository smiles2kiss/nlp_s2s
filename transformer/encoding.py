import os
import sys
import numpy as np
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.d_hid = d_hid
        self.n_position = n_position
        self.pos_tab = self._get_sinusoid_encoding_table(self.d_hid, self.n_position)

    def get_position_angle_vec(self, d_hid, position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    def _get_sinusoid_encoding_table(self, d_hid, n_position):
        sinusoid_table = [self.get_position_angle_vec(d_hid, pos_i) for pos_i in range(n_position)]
        sinusoid_table = np.array(sinusoid_table)
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # return torch.FloatTensor(sinusoid_table).unsqueeze(0)
        return torch.from_numpy(sinusoid_table).float().unsqueeze(0)

    def forward(self, src_seq):
        batch_size = src_seq.size(0)
        seq_len    = src_seq.size(1)
        pos_emb = self.pos_tab[:, :seq_len]
        return pos_emb
