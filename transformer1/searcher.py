import torch
import spacy
import torch.nn as nn
import torch.nn.functional as F

from transformer1.mask import make_src_mask
from transformer1.mask import make_trg_mask

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

    src_mask = make_src_mask(src_tensor, src_pad_idx)
    with torch.no_grad():
        enc_src = model.encoder(src=src_tensor, src_mask=src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        # trg_tensor: [1, trg_len]
        # trg_len:    [1]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
        trg_len    = torch.LongTensor([len(trg_indexes)]).cuda()

        trg_mask = make_trg_mask(trg_tensor, trg_pad_idx)

        with torch.no_grad():
            output, attention = model.decoder(trg=trg_tensor, enc_src=enc_src,
                                           trg_mask=trg_mask, src_mask=src_mask)

        pred_token = output.argmax(2)[:, -1].item()
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
    # src_mask:   [1, 1, 1, src_len]
    src_tensor = input_ids.unsqueeze(0).long().cuda()
    src_len    = torch.LongTensor([input_ids.size(0)]).cuda()
    src_mask   = make_src_mask(src_tensor, src_pad_idx)
    with torch.no_grad():
        enc_src = model.encoder(src=src_tensor, src_mask=src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        # trg_tensor: [1, trg_len]
        # trg_len:    [1]
        # trg_mask:   [1, 1, trg_len, trg_len]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
        trg_len    = torch.LongTensor([len(trg_indexes)]).cuda()
        trg_mask   = make_trg_mask(trg_tensor, trg_pad_idx)

        with torch.no_grad():
            output, attention = model.decoder(trg=trg_tensor, enc_src=enc_src, trg_mask=trg_mask, src_mask=src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]
