import torch
import spacy
import torch.nn as nn
import torch.nn.functional as F

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

    # src_tensor: [1, src_len]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).cuda()

    with torch.no_grad():
        encoded_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoded_conved, encoder_combined)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]: break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens


def translate_tokens(input_ids, src_field, trg_field, model, max_len=32):
    model.eval()

    # src_tensor: [1, src_len]
    src_tensor = input_ids.unsqueeze(0).long().cuda()

    # encoded_conved:   [batch_size, src_len, emb_dim]
    # encoder_combined: [batch_size, src_len, emb_dim]
    with torch.no_grad():
        encoded_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        # trg_tensor: [1, trg_len]
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()

        # output:    [1, trg_len, output_dim]
        # attention: [1, trg_len, src_len]
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoded_conved, encoder_combined)

        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]: break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]
