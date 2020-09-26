import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class Scaled_Dot_Product_Attention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, mask, scale):
        """
        :param Q: [batch_size, num_head, len_q, dim_q]
        :param K: [batch_size, num_head, len_k, dim_k]
        :param V: [batch_size, num_head, len_v, dim_v]
        :param scale: 缩放因子, dim_k
        :return:
        """
        # K: [batch_size, num_head, dim_k, len_k]
        K = K.permute(0, 1, 3, 2)

        # mm:     二维矩阵乘法，即 (n*m)   和 (m*q)
        # bmm:    三维张量相乘，即 (b*n*m) 和 (b*m*q)
        # matmul: 任意维度张量乘法

        # attention: [batch_size, num_head, len_q, len_k]
        attention = torch.matmul(Q, K)
        attention = attention * scale
        if mask is not None:
            attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)

        # context: [batch_size, num_head, len_q, dim_v]
        context = torch.matmul(attention, V)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, hidden_size=512, d_q=64, d_k=64, d_v=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        attn_size = hidden_size // num_head
        self.num_head    = num_head
        self.hidden_size = hidden_size
        self.attn_size   = attn_size  # 根号d

        self.scale = attn_size ** (-0.5)

        self.linear_q = nn.Linear(hidden_size, num_head * attn_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, num_head * attn_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, num_head * attn_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.attention = Scaled_Dot_Product_Attention()

        self.attn_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head * attn_size, hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        initialize_weight(self.output)

    def forward(self, q, k, v, mask):
        d_q = self.attn_size
        d_k = self.attn_size
        d_v = self.attn_size

        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)

        batch_size = q.size(0)
        residual = q

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)

        # q: [batch_size, len_q, num_head, attn_size]
        # k: [batch_size, len_k, num_head, attn_size]
        # v: [batch_size, len_v, num_head, attn_size]
        q = self.linear_q(q).view(batch_size, len_q, self.num_head, d_q)
        k = self.linear_k(k).view(batch_size, len_k, self.num_head, d_k)
        v = self.linear_v(v).view(batch_size, len_v, self.num_head, d_v)

        # q: [batch_size, num_head, len_q, attn_size]
        # k: [batch_size, num_head, len_k, attn_size]
        # v: [batch_size, num_head, len_v, attn_size]
        q = q.permute(0, 2, 1, 3)  # q = q.transpose(1, 2)
        k = k.permute(0, 2, 1, 3)  # k = k.transpose(1, 2)
        v = v.permute(0, 2, 1, 3)  # v = v.transpose(1, 2)

        # mask: [batch_size, len_q, len_k]
        # mask: [batch_size, num_head, len_q, len_k]
        mask = mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)

        # context:   [batch_size, num_head, len_v, attn_size]
        # attention: [batch_size, num_head, len_q, len_k]
        context, attention = self.attention(q, k, v, mask, self.scale)

        # context: [batch_size, len_v, num_head, attn_size]
        context = context.transpose(1, 2)
        context = context.contiguous().view(batch_size, len_q, self.num_head * self.attn_size)
        context = self.attn_dropout(context)

        context = context + residual
        context = self.layer_norm(context)
        return context, attention

