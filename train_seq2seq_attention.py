import torch
import torch.nn as nn
import torch.optim as optim

import os
import json
import time
import spacy
import random
import numpy as np

from torchtext.data import Field
from torchtext.data import BucketIterator
from torchtext.datasets import Multi30k
from seq2seq_attention.seq2seq import Seq2seq
from seq2seq_attention.searcher import Greedy_Search
from seq2seq_attention.searcher import Beam_Search

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def init_seed():
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def prepare_data(type="train"):
    SRC = Field(tokenize=tokenize_de, unk_token='<unk>', pad_token='<pad>', init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
    TGT = Field(tokenize=tokenize_en, unk_token='<unk>', pad_token='<pad>', init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TGT))
    print("train examples number", len(train_data.examples))
    print("valid examples number", len(valid_data.examples))
    print("test  examples number", len( test_data.examples))
    print(vars(train_data.examples[0]))
    print(vars(valid_data.examples[0]))

    SRC.build_vocab(train_data, min_freq=2, max_size=25000, vectors='glove.6B.300d')
    TGT.build_vocab(train_data, min_freq=2, max_size=25000, vectors='glove.6B.300d')
    print("src_vocab size = ", len(SRC.vocab))
    print("tgt_vocab size = ", len(TGT.vocab))

    train_batch_size = 128
    dev_batch_size   = 64
    device = torch.device('cuda')
    train_iterator = BucketIterator(train_data, batch_size=train_batch_size, device=device, shuffle=True)
    valid_iterator = BucketIterator(valid_data, batch_size=dev_batch_size,   device=device, shuffle=False)
    test_iterator  = BucketIterator(test_data,  batch_size=dev_batch_size,   device=device, shuffle=False)

    enc_emb_dim, enc_hid_dim, enc_n_layer, enc_dropout, input_dim  = 300, 300, 1, 0.1, len(SRC.vocab)
    dec_emb_dim, dec_hid_dim, dec_n_layer, dec_dropout, output_dim = 300, 300, 1, 0.1, len(TGT.vocab)
    attn_dim = 80

    src_unk_idx = SRC.vocab.stoi[SRC.unk_token]
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    src_sos_idx = SRC.vocab.stoi[SRC.init_token]
    src_eos_idx = SRC.vocab.stoi[SRC.eos_token]
    SRC.vocab.vectors.data[src_unk_idx] = torch.zeros(enc_emb_dim)
    SRC.vocab.vectors.data[src_pad_idx] = torch.zeros(enc_emb_dim)
    enc_embeddings = SRC.vocab.vectors

    tgt_unk_idx = TGT.vocab.stoi[TGT.unk_token]
    tgt_pad_idx = TGT.vocab.stoi[TGT.pad_token]
    tgt_sos_idx = TGT.vocab.stoi[TGT.init_token]
    tgt_eos_idx = TGT.vocab.stoi[TGT.eos_token]
    TGT.vocab.vectors.data[tgt_unk_idx] = torch.zeros(dec_emb_dim)
    TGT.vocab.vectors.data[tgt_pad_idx] = torch.zeros(dec_emb_dim)
    dec_embeddings = TGT.vocab.vectors

    if type in ["train"]:
        model = Seq2seq(enc_embeddings=enc_embeddings, dec_embeddings=dec_embeddings,
                        input_dim=input_dim,           output_dim=output_dim,
                        enc_emb_dim=enc_emb_dim,       dec_emb_dim=dec_emb_dim,
                        enc_hid_dim=enc_hid_dim,       dec_hid_dim=dec_hid_dim,
                        attn_dim=attn_dim,
                        enc_n_layer=enc_n_layer,       dec_n_layer=dec_n_layer,
                        enc_dropout=enc_dropout,       dec_dropout=dec_dropout)
    else:
        # model = Greedy_Search(enc_embeddings=enc_embeddings, dec_embeddings=dec_embeddings,
        #                       input_dim=input_dim,           output_dim=output_dim,
        #                       enc_emb_dim=enc_emb_dim,       dec_emb_dim=dec_emb_dim,
        #                       enc_hid_dim=enc_hid_dim,       dec_hid_dim=dec_hid_dim,
        #                       attn_dim=attn_dim,
        #                       enc_n_layer=enc_n_layer,       dec_n_layer=dec_n_layer,
        #                       enc_dropout=enc_dropout,       dec_dropout=dec_dropout)
        model = Beam_Search(enc_embeddings=enc_embeddings, dec_embeddings=dec_embeddings,
                            input_dim=input_dim,           output_dim=output_dim,
                            enc_emb_dim=enc_emb_dim,       dec_emb_dim=dec_emb_dim,
                            enc_hid_dim=enc_hid_dim,       dec_hid_dim=dec_hid_dim,
                            attn_dim=attn_dim,
                            enc_n_layer=enc_n_layer,       dec_n_layer=dec_n_layer,
                            enc_dropout=enc_dropout,       dec_dropout=dec_dropout)

    model.apply(init_weights)
    model.cuda()
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fct = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    return model, optimizer, loss_fct, train_iterator, valid_iterator, test_iterator, SRC, TGT


def train(model, optimizer, train_iterator, loss_fct):
    model.train()
    epoch_loss = 0
    for idx, batch in enumerate(train_iterator):
        # src: [seq_len, batch_size]
        # tgt: [seq_len, batch_size]
        src, src_len = batch.src
        tgt, tgt_len = batch.trg

        optimizer.zero_grad()
        # output: [seq_len, batch_size, output_dim]
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        tgt    = tgt[1:].view(-1)

        loss = loss_fct(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_iterator)


def eval(model, valid_iterator, loss_fct):
    model.eval()
    epoch_loss = 0

    for idx, batch in enumerate(valid_iterator):
        src, src_len = batch.src
        tgt, tgt_len = batch.trg
        with torch.no_grad():
            output = model(src, tgt, teacher_forcing_ratio=0)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].view(-1)

        loss = loss_fct(output, tgt)
        epoch_loss += loss.item()
    return epoch_loss / len(valid_iterator)


def do_train():
    init_seed()
    num_epoch = 100

    model, optimizer, loss_fct, train_iterator, valid_iterator, test_iterator, SRC, TGT = prepare_data(type="train")
    for epoch in range(num_epoch):
        train_loss = train(model, optimizer, train_iterator, loss_fct)
        valid_loss = eval( model, valid_iterator, loss_fct)

        model_dir  = "./checkpoint/seq2seq_attention"
        model_path = os.path.join(model_dir, f"model_{epoch}.pt")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)

        print("[TIME] --- time: {} --- [TIME]".format(time.ctime(time.time())))
        print("epoch: {}, train_loss: {}, valid_loss: {}".format(epoch, train_loss, valid_loss))


def do_predict():
    model, optimizer, loss_fct, train_iterator, valid_iterator, test_iterator, SRC, TGT = prepare_data(type="test")
    model_dir  = "./checkpoint/seq2seq_attention"
    model_path = os.path.join(model_dir, "model_9.pt")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    pre_sents = []
    gth_sents = []
    for idx, batch in enumerate(test_iterator):
        src, src_len = batch.src
        tgt, tgt_len = batch.trg

        with torch.no_grad():
            all_tokens, all_scores = model(src)

        # all_tokens: [batch_size, seq_len]
        # tgt:        [batch_size, seq_len]
        pre_tokens = all_tokens.cpu().detach().numpy().tolist()
        gth_tokens = tgt.transpose(0, 1).cpu().detach().numpy().tolist()
        for pre_ids, gth_ids in zip(pre_tokens, gth_tokens):
            pre = [TGT.vocab.itos[idx] for idx in pre_ids]
            gth = [TGT.vocab.itos[idx] for idx in gth_ids]
            pre_sents.append(" ".join(pre))
            gth_sents.append(" ".join(gth))

    pre_path = os.path.join(model_dir, "pre.json")
    gth_path = os.path.join(model_dir, "gth.json")
    with open(pre_path, "w", encoding="utf-8") as writer:
        json.dump(pre_sents, writer, ensure_ascii=False, indent=4)
    with open(gth_path, "w", encoding="utf-8") as writer:
        json.dump(gth_sents, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # do_train()
    do_predict()
