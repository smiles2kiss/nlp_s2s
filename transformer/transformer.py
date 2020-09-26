import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.mask import get_non_pad_mask
from transformer.mask import get_enc_attn_mask
from transformer.mask import get_dec_attn_mask
from transformer.mask import get_enc_dec_attn_mask


class Transformer(nn.Module):
    def __init__(self,
                 src_embeddings, trg_embeddings,
                 n_src_vocab, n_trg_vocab,
                 src_pad_idx, trg_pad_idx,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layer=6, n_head=8,
                 d_q=64, d_k=64, d_v=64,
                 dropout=0.1, n_position=200, max_seq_len=32):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(
            embeddings=src_embeddings,
            n_src_vocab=n_src_vocab,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_layer,
            n_head=n_head,
            d_q=d_q,
            d_k=d_k,
            d_v=d_v,
            pad_idx=src_pad_idx,
            dropout=dropout,
            n_position=n_position,
            max_seq_len=32
        )
        self.decoder = Decoder(
            embeddings=trg_embeddings,
            n_trg_vocab=n_trg_vocab,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_layer,
            n_head=n_head,
            d_q=d_q,
            d_k=d_k,
            d_v=d_v,
            pad_idx=trg_pad_idx,
            n_position=n_position,
            dropout=dropout
        )

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        # for name, param in self.named_parameters():
        #     if param.dim() > 1:
        #         nn.init.xavier_normal(param)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, src_seq, src_len, trg_seq, trg_len):
        src_pad_mask  = get_non_pad_mask( src_seq, self.src_pad_idx).cuda()
        src_attn_mask = get_enc_attn_mask(src_seq, self.src_pad_idx).cuda()

        trg_pad_mask  = get_non_pad_mask( trg_seq, self.trg_pad_idx).cuda()
        trg_attn_mask = get_dec_attn_mask(trg_seq, self.trg_pad_idx).cuda()

        trg_src_attn_mask = get_enc_dec_attn_mask(src_seq, src_len, trg_seq, trg_len, self.trg_pad_idx).cuda()

        enc_output, *_ = self.encoder(src_seq=src_seq, src_pad_mask=src_pad_mask, src_attn_mask=src_attn_mask)
        dec_output, *_ = self.decoder(trg_seq=trg_seq, trg_pad_mask=trg_pad_mask, trg_attn_mask=trg_attn_mask, enc_output=enc_output, dec_enc_attn_mask=trg_src_attn_mask)

        # dec_output: [batch_size, seq_len, trg_vocab_size]
        seq_logit = self.trg_word_prj(dec_output)
        seq_logit = seq_logit.view(-1, seq_logit.size(2))
        return seq_logit
