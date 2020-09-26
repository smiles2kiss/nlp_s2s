import os
import sys
import json
import time
import spacy
import torch
import torch.nn as nn
from torch.optim import Adam

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import BucketIterator
from torchtext.datasets import Multi30k
from torchtext.datasets import WMT14
from torchtext.datasets import TranslationDataset
from transformer1.transformer import Transformer
from transformer1.searcher import translate_tokens

spacy_zh = spacy.load('zh_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def prepare_data_multi30k():
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    SRC = Field(tokenize=tokenize_de,
                unk_token=UNK_TOKEN,
                pad_token=PAD_TOKEN,
                init_token=SOS_TOKEN,
                eos_token=EOS_TOKEN,
                fix_length=32,
                lower=True,
                batch_first=True,
                include_lengths=True)
    TGT = Field(tokenize=tokenize_en,
                unk_token=UNK_TOKEN,
                pad_token=PAD_TOKEN,
                init_token=SOS_TOKEN,
                eos_token=EOS_TOKEN,
                fix_length=32,
                lower=True,
                batch_first=True,
                include_lengths=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TGT))
    SRC.build_vocab(train_data, min_freq=2, max_size=25000)
    TGT.build_vocab(train_data, min_freq=2, max_size=25000)

    train_batch_size = 64
    dev_batch_size   = 64
    device = torch.device('cuda')
    train_iterator = BucketIterator(train_data, batch_size=train_batch_size, device=device, shuffle=True)
    valid_iterator = BucketIterator(valid_data, batch_size=dev_batch_size,   device=device, shuffle=False)
    test_iterator  = BucketIterator(test_data,  batch_size=dev_batch_size,   device=device, shuffle=False)

    src_unk_idx = SRC.vocab.stoi[SRC.unk_token]
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    src_sos_idx = SRC.vocab.stoi[SRC.init_token]
    src_eos_idx = SRC.vocab.stoi[SRC.eos_token]
    src_vocab_size = len(SRC.vocab)

    tgt_unk_idx = TGT.vocab.stoi[TGT.unk_token]
    tgt_pad_idx = TGT.vocab.stoi[TGT.pad_token]
    tgt_sos_idx = TGT.vocab.stoi[TGT.init_token]
    tgt_eos_idx = TGT.vocab.stoi[TGT.eos_token]
    tgt_vocab_size = len(TGT.vocab)

    return train_iterator, valid_iterator, test_iterator,\
           src_vocab_size, tgt_vocab_size, \
           src_pad_idx, tgt_pad_idx, SRC, TGT


def prepare_data_wmt14():
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    MAX_LEN = 32
    MIN_FREQ = 2
    LOWER = True
    SRC = Field(tokenize=tokenize_en, batch_first=True, lower=LOWER, include_lengths=True, fix_length=MAX_LEN,
                unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
    TRG = Field(tokenize=tokenize_de, batch_first=True, lower=LOWER, include_lengths=True, fix_length=MAX_LEN,
                unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)
    train_data, valid_data, test_data = WMT14.splits(exts=('.en', '.de'), fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=MIN_FREQ, max_size=25000, vectors='glove.6B.300d')
    TRG.build_vocab(train_data, min_freq=MIN_FREQ, max_size=25000, vectors='glove.6B.300d')

    enc_emb_dim = 256
    dec_emb_dim = 256
    train_batch_size = 128
    dev_batch_size   = 64
    device = torch.device('cuda')
    train_iterator = BucketIterator(train_data, batch_size=train_batch_size, device=device, shuffle=True)
    valid_iterator = BucketIterator(valid_data, batch_size=dev_batch_size,   device=device, shuffle=False)
    test_iterator  = BucketIterator(test_data,  batch_size=dev_batch_size,   device=device, shuffle=False)

    src_unk_idx = SRC.vocab.stoi[SRC.unk_token]
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    src_sos_idx = SRC.vocab.stoi[SRC.init_token]
    src_eos_idx = SRC.vocab.stoi[SRC.eos_token]
    SRC.vocab.vectors.data[src_unk_idx] = torch.zeros(enc_emb_dim)
    SRC.vocab.vectors.data[src_pad_idx] = torch.zeros(enc_emb_dim)
    src_embeddings = SRC.vocab.vectors

    tgt_unk_idx = TRG.vocab.stoi[TRG.unk_token]
    tgt_pad_idx = TRG.vocab.stoi[TRG.pad_token]
    tgt_sos_idx = TRG.vocab.stoi[TRG.init_token]
    tgt_eos_idx = TRG.vocab.stoi[TRG.eos_token]
    TRG.vocab.vectors.data[tgt_unk_idx] = torch.zeros(dec_emb_dim)
    TRG.vocab.vectors.data[tgt_pad_idx] = torch.zeros(dec_emb_dim)
    dec_embeddings = TRG.vocab.vectors
    return train_iterator, valid_iterator, test_iterator, src_embeddings, dec_embeddings, SRC, TRG


def train_epoch(model, optimizer, train_iterator, trg_pad_idx, smoothing=False):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    total_loss = 0
    total_step = 0
    for batch in train_iterator:
        # src_seq: [batch_size, src_len]
        # tgt_seq: [batch_size, tgt_len]
        src_seq, src_len = batch.src
        tgt_seq, tgt_len = batch.trg
        src_seq = src_seq.cuda()
        tgt_seq = tgt_seq.cuda()
        assert src_seq.size(0) == tgt_seq.size(0)

        optimizer.zero_grad()
        # output: [batch_size, trg_len, output_dim]
        output, _ = model(src_seq, tgt_seq[:, :-1])

        output_dim = output.size(-1)

        # output:  [(batch_size*(trg_len-1), output_dim]
        # tgt_seq: [(batch_size*(trg_len-1)]
        output = output.contiguous().view(-1, output_dim)
        tgt_seq = tgt_seq[:, 1:].contiguous().view(-1)

        loss = criterion(input=output, target=tgt_seq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_step += 1
        total_loss += loss.item()

        if total_step % 100 == 0:
            print("[TIME] --- time: {} --- [TIME], total_step: {}".format(time.ctime(time.time()), total_step))
    return total_loss / total_step


def eval_epoch(model, valid_iterator, trg_pad_idx, smoothing=False):
    model.eval()
    total_loss = 0
    total_step = 0
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    for batch in valid_iterator:
        # src_seq: [batch_size, src_len]
        # tgt_seq: [batch_size, tgt_len]
        src_seq, src_len = batch.src
        tgt_seq, tgt_len = batch.trg
        src_seq = src_seq.cuda()
        tgt_seq = tgt_seq.cuda()

        with torch.no_grad():
            output, _ = model(src_seq, tgt_seq[:, :-1])

        output_dim = output.size(-1)

        # output:  [(batch_size*(trg_len-1)), output_dim]
        # tgt_seq: [(batch_size*(trg_len-1))]
        output = output.contiguous().view(-1, output_dim)
        tgt_seq = tgt_seq[:, 1:].contiguous().view(-1)

        loss = criterion(input=output, target=tgt_seq)

        total_step += 1
        total_loss += loss.item()

    return total_loss / total_step


def do_train():
    train_iterator, valid_iterator, test_iterator, \
    src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, SRC, TGT = prepare_data_multi30k()
    model = Transformer(enc_input_dim=src_vocab_size, dec_output_dim=tgt_vocab_size,
                        enc_hid_dim=256,              dec_hid_dim=256,
                        enc_n_layer=3,                dec_n_layer=3,
                        enc_n_head=8,                 dec_n_head=8,
                        enc_pf_dim=512,               dec_pf_dim=512,
                        enc_dropout=0.1,              dec_dropout=0.1,
                        enc_max_length=32,            dec_max_length=32,
                        src_pad_idx=src_pad_idx,      trg_pad_idx=tgt_pad_idx)

    model.cuda()
    optimizer = Adam(model.parameters(), lr=5e-4)

    num_epoch = 10
    for epoch in range(num_epoch):
        train_loss = train_epoch(model, optimizer, train_iterator, tgt_pad_idx, smoothing=False)
        eval_loss  = eval_epoch(model, valid_iterator, tgt_pad_idx, smoothing=False)

        model_dir  = os.path.join("./checkpoint/transformer1")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        print("[TIME] --- {} --- [TIME]".format(time.ctime(time.time())))
        print("epoch: {}, train_loss: {}, eval_loss: {}".format(epoch, train_loss, eval_loss))


def do_predict():
    train_iterator, valid_iterator, test_iterator, \
    src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, SRC, TGT = prepare_data_multi30k()

    model = Transformer(enc_input_dim=src_vocab_size, dec_output_dim=tgt_vocab_size,
                        enc_hid_dim=256,              dec_hid_dim=256,
                        enc_n_layer=3,                dec_n_layer=3,
                        enc_n_head=8,                 dec_n_head=8,
                        enc_pf_dim=512,               dec_pf_dim=512,
                        enc_dropout=0.1,              dec_dropout=0.1,
                        enc_max_length=32,            dec_max_length=32,
                        src_pad_idx=src_pad_idx,      trg_pad_idx=tgt_pad_idx)
    model.cuda()

    model_dir  = "./checkpoint/transformer1"
    model_path = os.path.join(model_dir, "model_9.pt")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    pre_sents = []
    gth_sents = []
    for idx, batch in enumerate(test_iterator):
        # src_seq: [batch_size, src_len]
        # tgt_seq: [batch_size, tgt_len]
        src_seq, src_len = batch.src
        tgt_seq, tgt_len = batch.trg
        src_seq = src_seq.cuda()
        tgt_seq = tgt_seq.cuda()

        batch_size = src_seq.size(0)
        pre_tokens = []
        with torch.no_grad():
            for idx in range(batch_size):
                tokens = translate_tokens(src_seq[idx], SRC, TGT, model, max_len=32)
                pre_tokens.append(tokens)

        # tgt: [batch_size, seq_len]
        gth_tokens = tgt_seq.cpu().detach().numpy().tolist()
        for tokens, gth_ids in zip(pre_tokens, gth_tokens):
            gth = [TGT.vocab.itos[idx] for idx in gth_ids[1:]]
            pre_sents.append(" ".join(tokens))
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
