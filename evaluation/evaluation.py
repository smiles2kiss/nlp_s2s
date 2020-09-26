import os
import sys
import json
import spacy
from torchtext.data.metrics import bleu_score
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)

# model_dir = os.path.join(par_dir, "checkpoint/seq2seq")
model_dir = os.path.join(par_dir, "checkpoint/seq2seq_attention")
pre_path = os.path.join(model_dir, "pre.json")
gth_path = os.path.join(model_dir, "gth.json")

with open(pre_path, "r", encoding="utf-8") as reader:
    input_pre = json.load(reader)
with open(gth_path, "r", encoding="utf-8") as reader:
    input_gth = json.load(reader)

candidate_corpus = []
references_corpus = []
for pre, gth in zip(input_pre, input_gth):
    pre = pre.replace("<sos>", "").replace("<eos>", "").replace("<unk>", "unk").strip()
    gth = gth.replace("<sos>", "").replace("<eos>", "").replace("<unk>", "unk").strip()
    pre_words = tokenize_en(pre)
    gth_words = tokenize_en(gth)

    # pre_words = pre_words[:32] + ["<unk>"] * (32 - len(pre_words))
    # gth_words = gth_words[:32] + ["<unk>"] * (32 - len(gth_words))
    pre_words = pre_words if len(pre_words) > 4 else ["unk"] * 4
    gth_words = gth_words if len(gth_words) > 4 else ["unk"] * 4
    candidate_corpus.append(pre_words)
    references_corpus.append(gth_words)

score = bleu_score(candidate_corpus, references_corpus)
print("bleu_score = ", score)
