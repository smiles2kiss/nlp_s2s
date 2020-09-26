import torch
import spacy
import torch.nn as nn
import torch.nn.functional as F

from transformer.mask import get_non_pad_mask
from transformer.mask import get_enc_attn_mask
from transformer.mask import get_dec_attn_mask
from transformer.mask import get_enc_dec_attn_mask

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text.lowe() for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


def translate_sentence(sentence, src_field, trg_field, model, max_len=32):
    model.eval()
    tokens = tokenize_de(sentence)

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
    trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]

    # src_tensor: [1, src_len]
    # src_len:    [1]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).cuda()
    src_len    = torch.LongTensor([len(src_indexes)]).cuda()

    src_pad_mask  = get_non_pad_mask( src_tensor, src_pad_idx).cuda()
    src_attn_mask = get_enc_attn_mask(src_tensor, src_pad_idx).cuda()

    with torch.no_grad():
        enc_output, *_ = model.encoder(src_seq=src_tensor, src_pad_mask=src_pad_mask, src_attn_mask=src_attn_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        # trg_tensor: [1, trg_len]
        # trg_len:    [1]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
        trg_len    = torch.LongTensor([len(trg_indexes)]).cuda()

        trg_pad_mask  = get_non_pad_mask( trg_tensor, trg_pad_idx).cuda()
        trg_attn_mask = get_dec_attn_mask(trg_tensor, trg_pad_idx).cuda()

        trg_src_attn_mask = get_enc_dec_attn_mask(src_tensor, src_len, trg_tensor, trg_len, trg_pad_idx).cuda()

        with torch.no_grad():
            dec_output, *_ = model.decoder(trg_seq=trg_tensor, trg_pad_mask=trg_pad_mask, trg_attn_mask=trg_attn_mask,
                                           enc_output=enc_output, dec_enc_attn_mask=trg_src_attn_mask)
        # seq_logit: [batch_size, seq_len, trg_vocab_size]
        seq_logit = model.trg_word_prj(dec_output)
        pred_token = dec_output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]


def translate_tokens(input_ids, src_field, trg_field, model, max_len=32):
    model.eval()

    src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
    trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]

    # src_tensor: [1, src_len]
    # src_len:    [1]
    src_tensor = input_ids.long().unsqueeze(0).cuda()
    src_len = torch.LongTensor([input_ids.size(0)]).cuda()

    src_pad_mask  = get_non_pad_mask(src_tensor, src_pad_idx).cuda()
    src_attn_mask = get_enc_attn_mask(src_tensor, src_pad_idx).cuda()

    with torch.no_grad():
        enc_output, *_ = model.encoder(src_seq=src_tensor, src_pad_mask=src_pad_mask, src_attn_mask=src_attn_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        # trg_tensor: [1, trg_len]
        # trg_len:    [1]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
        trg_len = torch.LongTensor([len(trg_indexes)]).cuda()

        trg_pad_mask  = get_non_pad_mask(trg_tensor, trg_pad_idx).cuda()
        trg_attn_mask = get_dec_attn_mask(trg_tensor, trg_pad_idx).cuda()
        trg_src_attn_mask = get_enc_dec_attn_mask(src_tensor, src_len, trg_tensor, trg_len, trg_pad_idx).cuda()

        with torch.no_grad():
            dec_output, *_ = model.decoder(trg_seq=trg_tensor, trg_pad_mask=trg_pad_mask, trg_attn_mask=trg_attn_mask,
                                           enc_output=enc_output, dec_enc_attn_mask=trg_src_attn_mask)
        # seq_logit: [batch_size, seq_len, trg_vocab_size]
        seq_logit = model.trg_word_prj(dec_output)
        pred_token = seq_logit.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens
