import os
import sys
import json
import time
import spacy
import torch
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
from transformer.transformer import Transformer
from transformer.evaluate import cal_performance
from transformer.searcher import translate_tokens

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
    SRC.build_vocab(train_data, min_freq=2, max_size=25000, vectors='glove.6B.300d')
    TGT.build_vocab(train_data, min_freq=2, max_size=25000, vectors='glove.6B.300d')

    enc_emb_dim = 300
    dec_emb_dim = 300
    train_batch_size = 32
    dev_batch_size   = 16
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
    src_vocab_size = len(SRC.vocab)

    tgt_unk_idx = TGT.vocab.stoi[TGT.unk_token]
    tgt_pad_idx = TGT.vocab.stoi[TGT.pad_token]
    tgt_sos_idx = TGT.vocab.stoi[TGT.init_token]
    tgt_eos_idx = TGT.vocab.stoi[TGT.eos_token]
    TGT.vocab.vectors.data[tgt_unk_idx] = torch.zeros(dec_emb_dim)
    TGT.vocab.vectors.data[tgt_pad_idx] = torch.zeros(dec_emb_dim)
    trg_embeddings = TGT.vocab.vectors
    tgt_vocab_size = len(TGT.vocab)

    return train_iterator, valid_iterator, test_iterator,\
           src_embeddings, trg_embeddings, \
           src_vocab_size, tgt_vocab_size, \
           src_pad_idx, tgt_pad_idx, SRC, TGT


def prepare_data_wmt14():
    # SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
    # TGT = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, include_lengths=True)
    # train_data, valid_data, test_data = TranslationDataset.splits(exts=('.src', '.trg'), fields=(SRC, TGT))
    # SRC.build_vocab(train_data, min_freq=2, max_size=25000, vectors='glove.6B.300d')
    # TGT.build_vocab(train_data, min_freq=2, max_size=25000, vectors='glove.6B.300d')

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

    enc_emb_dim = 300
    dec_emb_dim = 300
    train_batch_size = 128
    dev_batch_size = 64
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
    total_loss = 0
    total_step = 0
    n_word_total = 0
    n_word_correct = 0
    for batch in train_iterator:
        # src_seq: [seq_len, batch_size]
        # tgt_seq: [seq_len, batch_size]
        src_seq, src_len = batch.src
        tgt_seq, tgt_len = batch.trg

        # src_seq: [batch_size, seq_len]
        # tgt_seq: [batch_size, seq_len]
        src_seq = src_seq.transpose(0, 1).cuda()
        tgt_seq = tgt_seq.transpose(0, 1).cuda()

        optimizer.zero_grad()
        pred = model(src_seq, src_len, tgt_seq, tgt_len)
        loss, n_correct, n_word = cal_performance(pred, tgt_seq, trg_pad_idx=trg_pad_idx, smoothing=smoothing)
        loss.backward()
        optimizer.step()

        n_word_total   += n_word
        n_word_correct += n_correct

        total_step += 1
        total_loss += loss.item()

        if total_step % 100 == 0:
            print("[TIME] --- time: {} --- [TIME], total_step: {}".format(time.ctime(time.time()), total_step))

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, valid_iterator, trg_pad_idx, smoothing=False):
    model.eval()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    for batch in valid_iterator:
        # src_seq: [seq_len, batch_size]
        # tgt_seq: [seq_len, batch_size]
        src_seq, src_len = batch.src
        tgt_seq, tgt_len = batch.trg

        # src_seq: [batch_size, seq_len]
        # tgt_seq: [batch_size, seq_len]
        src_seq = src_seq.transpose(0, 1).cuda()
        tgt_seq = tgt_seq.transpose(0, 1).cuda()

        with torch.no_grad():
            pred = model(src_seq, src_len, tgt_seq, tgt_len)
        loss, n_correct, n_word = cal_performance(pred, tgt_seq, trg_pad_idx=trg_pad_idx, smoothing=smoothing)

        n_word_total   += n_word
        n_word_correct += n_correct
        total_loss     += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy  = n_word_correct / n_word_total
    return loss_per_word, accuracy


def do_train():
    train_iterator, valid_iterator, test_iterator, src_embeddings, trg_embeddings, \
    src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, SRC, TGT = prepare_data_multi30k()
    model = Transformer(src_embeddings=src_embeddings, trg_embeddings=trg_embeddings,
                        n_src_vocab=src_vocab_size,    n_trg_vocab=tgt_vocab_size,
                        src_pad_idx=src_pad_idx,       trg_pad_idx=tgt_pad_idx,
                        d_word_vec=512, d_model=512, d_inner=2048,
                        n_layer=6, n_head=8, d_q=64, d_k=64, d_v=64,
                        dropout=0.1, n_position=200)

    model.cuda()
    optimizer = Adam(model.parameters(), lr=1e-3)

    num_epoch = 10
    for epoch in range(num_epoch):
        train_loss, train_accuracy = train_epoch(model, optimizer, train_iterator, tgt_pad_idx, smoothing=False)
        eval_loss,  eval_accuracy  = eval_epoch(model, valid_iterator, tgt_pad_idx, smoothing=False)

        model_dir  = os.path.join("./checkpoint/transformer")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        print("[TIME] --- {} --- [TIME]".format(time.ctime(time.time())))
        print("epoch: {}, train_loss: {}, eval_loss: {}".format(epoch, train_loss, eval_loss))
        print("epoch: {}, train_accuracy: {}, eval_accuracy: {}".format(epoch, train_accuracy, eval_accuracy))


def do_predict():
    train_iterator, valid_iterator, test_iterator, src_embeddings, trg_embeddings, \
    src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, SRC, TGT = prepare_data_multi30k()

    model = Transformer(src_embeddings=src_embeddings, trg_embeddings=trg_embeddings,
                        n_src_vocab=src_vocab_size, n_trg_vocab=tgt_vocab_size,
                        src_pad_idx=src_pad_idx, trg_pad_idx=tgt_pad_idx,
                        d_word_vec=512, d_model=512, d_inner=2048,
                        n_layer=6, n_head=8, d_q=64, d_k=64, d_v=64,
                        dropout=0.1, n_position=200)
    model.cuda()

    model_dir  = "./checkpoint/transformer"
    model_path = os.path.join(model_dir, "model_9.pt")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    model.eval()
    pre_sents = []
    gth_sents = []
    for idx, batch in enumerate(test_iterator):
        # src_seq: [seq_len, batch_size]
        # tgt_seq: [seq_len, batch_size]
        src_seq, src_len = batch.src
        tgt_seq, tgt_len = batch.trg

        # src_seq: [batch_size, seq_len]
        # tgt_seq: [batch_size, seq_len]
        src_seq = src_seq.transpose(0, 1).cuda()
        tgt_seq = tgt_seq.transpose(0, 1).cuda()

        batch_size = src_seq.size(0)
        pre_tokens = []
        with torch.no_grad():
            for idx in range(batch_size):
                tokens = translate_tokens(src_seq[idx], SRC, TGT, model, max_len=32)
                pre_tokens.append(tokens)

        # tgt: [batch_size, seq_len]
        gth_tokens = tgt_seq.cpu().detach().numpy().tolist()
        for tokens, gth_ids in zip(pre_tokens, gth_tokens):
            gth = [TGT.vocab.itos[idx] for idx in gth_ids]
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
