import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_head,
                 dropout):
        super(MultiHeadAttentionLayer, self).__init__()

        self.hid_dim = hid_dim
        self.n_head  = n_head
        self.head_dim = self.hid_dim // self.n_head  # d

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, query_len, hid_dim]
        # key:   [batch_size,   key_len, hid_dim]
        # value: [batch_size, value_len, hid_dim]

        batch_size = query.size(0)
        query_len  = query.size(1)
        key_len    = key.size(1)
        value_len  = value.size(1)

        # Q: [batch_size, query_len, hid_dim]
        # K: [batch_size,   key_len, hid_dim]
        # V: [batch_size, value_len, hid_dim]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, num_head, head_dim]
        # K: [batch_size,   key_len, num_head, head_dim]
        # V: [batch_size, value_len, num_head, head_dim]
        Q = Q.view(batch_size, query_len, self.n_head, self.head_dim)
        K = K.view(batch_size, key_len,   self.n_head, self.head_dim)
        V = V.view(batch_size, value_len, self.n_head, self.head_dim)

        # Q: [batch_size, num_head, query_len, head_dim]
        # K: [batch_size, num_head,   key_len, head_dim]
        # V: [batch_size, num_head, value_len, head_dim]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # energy: [batch_size, n_head, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        # attention: [batch_size, n_head, query_len, key_len]
        attention = torch.softmax(energy, dim=-1)

        # x: [batch_size, n_head, query_len, head_dim]
        x = torch.matmul(self.dropout(attention), V)

        # x: [batch_size, query_len, n_head, head_dim]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, query_len, hid_dim]
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
