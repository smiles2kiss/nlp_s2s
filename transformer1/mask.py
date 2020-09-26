import torch

def make_src_mask(src, src_pad_idx):
    # src: [batch_size, src_len]

    # src_mask: [batch_size, 1, 1, src_len]
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_trg_mask(trg, trg_pad_idx):
    # trg: [batch_size, trg_len]

    # trg_pad_mask: [batch_size, 1, 1, trg_len]
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)

    batch_size = trg.size(0)
    trg_len    = trg.size(1)

    # mask     ||  sub_mask
    # 1, 1, 1  ||  1, 0, 0
    # 1, 1, 1  ||  1, 1, 0
    # 1, 1, 1  ||  1, 1, 1

    # trg_sub_mask: [trg_len, trg_len]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).cuda()
    # trg_sub_mask: [1, 1, trg_len, trg_len]
    trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(0)
    # trg_sub_mask: [batch_size, 1, trg_len, trg_len]
    trg_sub_mask = trg_sub_mask.repeat(batch_size, 1, 1, 1)

    # trg_mask: [batch_size, 1, trg_len, trg_len]
    trg_mask = trg_pad_mask.byte() & trg_sub_mask.byte()
    return trg_mask
