import torch
import numpy as np


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).byte()
    return subsequent_mask

# def get_non_pad_mask(seq, pad_idx):
#     # seq: [batch_size, seq_len]
#     # out: [batch_size, 1, seq_len]
#     batch_size = seq.size(0)
#     seq_len    = seq.size(1)
#     pad_mask = seq.ne(pad_idx)
#     pad_mask = pad_mask.unsqueeze(1)
#     return pad_mask
#
#
# def get_enc_attn_mask(seq_q, pad_idx):
#     batch_size = seq_q.size(0)
#     seq_len    = seq_q.size(1)
#
#     pad_attn_mask = seq_q.ne(pad_idx)
#     pad_attn_mask = pad_attn_mask.unsqueeze(1)
#     pad_attn_mask = pad_attn_mask.expand(batch_size, seq_len, seq_len)
#     return pad_attn_mask
#
#
# def get_dec_attn_mask(seq_k, pad_idx):
#     batch_size = seq_k.size(0)
#     seq_len    = seq_k.size(1)
#
#     pad_attn_mask = seq_k.ne(pad_idx)
#     pad_attn_mask = pad_attn_mask.unsqueeze(1)
#     pad_attn_mask = pad_attn_mask.expand(batch_size, seq_len, seq_len)
#
#     subsequent_mask = get_subsequent_mask(seq_k, pad_idx)
#     pad_attn_mask = (pad_attn_mask.byte() + subsequent_mask.byte()).eq(2)
#     # pad_attn_mask = pad_attn_mask & subsequent_mask
#     return pad_attn_mask
#
#
# def get_subsequent_mask(seq_k, pad_idx):
#     batch_size = seq_k.size(0)
#     seq_len    = seq_k.size(1)
#
#     subsequent_mask = torch.ones((batch_size, seq_len, seq_len))
#     subsequent_mask = torch.triu(subsequent_mask, diagonal=1)
#     subsequent_mask = 1 - subsequent_mask
#     subsequent_mask = subsequent_mask.byte().cuda()
#     return subsequent_mask
#
#
# def get_enc_dec_attn_mask(seq_q, src_len, seq_k, tgt_len, pad_idx):
#     max_seq_q_len = torch.max(src_len).item()
#     max_seq_k_len = torch.max(tgt_len).item()
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#
#     # pad_attn_mask: [batch_size, len_k, len_q]
#     pad_attn_mask = torch.zeros(len_k, len_q)
#     pad_attn_mask[:max_seq_k_len, :max_seq_q_len] = 1
#     pad_attn_mask = pad_attn_mask.unsqueeze(0)
#     pad_attn_mask = pad_attn_mask.expand(batch_size, len_k, len_q)
#     return pad_attn_mask
