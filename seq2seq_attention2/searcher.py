import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from queue import Queue
from queue import PriorityQueue
from seq2seq_attention2.encoder import Encoder
from seq2seq_attention2.decoder import Decoder

unk_idx = 0
pad_idx = 1
sos_idx = 2
eos_idx = 3


class Greedy_Search(nn.Module):
    def __init__(self,
                 enc_embeddings, dec_embeddings,
                 input_size,   output_size,
                 enc_emb_size, dec_emb_size,
                 enc_hid_size, dec_hid_size,
                 enc_n_layer,  dec_n_layer,
                 enc_dropout,  dec_dropout):
        super(Greedy_Search, self).__init__()
        self.encoder = Encoder(enc_embeddings, input_size,  enc_emb_size, enc_hid_size, enc_n_layer, enc_dropout)
        self.decoder = Decoder(dec_embeddings, output_size, dec_emb_size, dec_hid_size, dec_n_layer, dec_dropout)
        self.sos_idx = sos_idx
        self.dec_n_layer = dec_n_layer

    def forward(self, input_ids, input_len):
        max_len = 32
        batch_size = input_ids.size(1)

        # encoder_outputs: [seq_len, batch_size, 2*hidden_size]
        # encoder_hidden:  [batch_size, hidden_size]
        # decoder_hidden:  [batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_ids, input_len)
        decoder_hidden = encoder_hidden

        # decoder_input: [batch_size]
        # all_tokens:    [batch_size, 1]
        # all_scores:    [batch_size, 1]
        sos_input = [self.sos_idx for _ in range(batch_size)]
        decoder_input = torch.LongTensor(sos_input).cuda()
        all_tokens = torch.LongTensor(sos_input).unsqueeze(1).long().cuda()
        all_scores = torch.LongTensor(sos_input).unsqueeze(1).long().cuda()

        for _ in range(max_len):
            # decoder_output: [batch_size, output_dim]
            # decoder_hidden: [batch_size, hidden_size]
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # decoder_scores: [batch_size, 1]
            # decoder_input:  [batch_size, 1]
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input.unsqueeze(1).long()),  dim=1)
            all_scores = torch.cat((all_scores, decoder_scores.unsqueeze(1).long()), dim=1)

        # all_tokens: [batch_size, max_len]
        # all_scores: [batch_size, max_len]
        return all_tokens, all_scores


class Node(object):
    def __init__(self, hidden, previous_node, decoder_input, log_prob, length):
        self.hidden = hidden  # 当前单词的hidden_state
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.log_prob = log_prob
        self.length = length


class Beam_Search(nn.Module):
    # Sequence to Sequence Learning with Neural Networks
    def __init__(self,
                 enc_embeddings, dec_embeddings,
                 input_size,   output_size,
                 enc_emb_size, dec_emb_size,
                 enc_hid_size, dec_hid_size,
                 enc_n_layer,  dec_n_layer,
                 enc_dropout,  dec_dropout):
        super(Beam_Search, self).__init__()
        self.encoder = Encoder(enc_embeddings, input_size,  enc_emb_size, enc_hid_size, enc_n_layer, enc_dropout)
        self.decoder = Decoder(dec_embeddings, output_size, dec_emb_size, dec_hid_size, dec_n_layer, dec_dropout)
        self.sos_idx = sos_idx
        self.dec_n_layer = dec_n_layer

    def forward(self, input_ids, input_len):
        batch_size = input_ids.size(1)
        max_len = 32

        sos_input = [self.sos_idx for _ in range(batch_size)]

        # decoder_input: [batch_size]
        # all_tokens:    [batch_size, 1]
        # all_scores:    [batch_size, 1]
        all_tokens = torch.LongTensor(batch_size, max_len).fill_(eos_idx).cuda()
        all_scores = torch.FloatTensor(batch_size, max_len).fill_(0).cuda()

        # encoder_outputs: [seq_len, batch_size, hidden_size]
        # encoder_hidden:  [batch_size, hidden_size]
        # decoder_hidden:  [batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_ids, input_len)
        decoder_hidden = encoder_hidden

        for batch_idx in range(batch_size):
            # _decoder_input: [1]
            # _encoder_outputs: [seq_len, 1, hidden_size]
            # _decoder_hidden:  [         1, hidden_size]
            _decoder_input = torch.LongTensor([sos_idx]).cuda()
            _encoder_outputs = encoder_outputs[:, batch_idx, :].unsqueeze(1)
            _decoder_hidden  = decoder_hidden[batch_idx, :].unsqueeze(0)

            root = Node(hidden=_decoder_hidden, previous_node=None, decoder_input=_decoder_input, log_prob=torch.FloatTensor([1]), length=1)
            q = Queue()
            q.put(root)

            end_nodes = []
            beam_size = 2
            while not q.empty():

                if len(end_nodes) >= beam_size:
                    break

                count = q.qsize()
                candidates = PriorityQueue()

                for idx in range(count):
                    node = q.get()

                    # _decoder_input: [1]
                    # _hidden: [1, hidden_size]
                    _decoder_input = node.decoder_input
                    _hidden        = node.hidden
                    _log_prob      = node.log_prob
                    _length        = node.length

                    if _decoder_input.item() == eos_idx or _length >= max_len:
                        end_nodes.append(node)
                        break

                    # decoder_output: [1, output_dim]
                    # decoder_hidden: [1, hidden_size]
                    _decoder_output, _decoder_hidden = self.decoder(_decoder_input.contiguous(), _hidden.contiguous(), _encoder_outputs.contiguous())

                    softmax = nn.Softmax()
                    decoder_score = softmax(_decoder_output)
                    values, indices = decoder_score.squeeze(0).topk(beam_size)

                    for i in range(beam_size):
                        prob = values[i] * _log_prob.item()
                        index = indices[i]
                        _decoder_input = torch.LongTensor([index]).cuda()
                        _len = _length + 1
                        new_node = Node(hidden=_decoder_hidden, previous_node=node, decoder_input=_decoder_input, log_prob=prob, length=_len)
                        candidates.put((prob, new_node))

                if candidates.qsize() == 0:
                    continue

                # 小根堆 转成 大根堆
                _candidates = []
                while not candidates.empty():
                    binode = candidates.get()
                    _candidates = [binode] + _candidates

                # 取topk = beam_size个元素
                for i in range(beam_size):
                    binode = _candidates[i]
                    q.put(binode[1])

            # 回溯: 找到整个句子的所有节点
            probs, indexs = torch.Tensor([node.log_prob for node in end_nodes]).topk(1)
            max_node = end_nodes[indexs[0]]

            _token_idxs   = []
            _token_scores = []
            while max_node is not None:
                _token_idxs   = [max_node.decoder_input.item()] + _token_idxs
                _token_scores = [max_node.log_prob.item()] + _token_scores
                max_node = max_node.previous_node

            token_len = len(_token_idxs)
            token_idxs   = torch.LongTensor(max_len).fill_(eos_idx)
            token_scores = torch.FloatTensor(max_len).fill_(0)
            token_idxs[:token_len]   = torch.LongTensor(_token_idxs).cuda()
            token_scores[:token_len] = torch.FloatTensor(_token_scores).cuda()
            all_tokens[batch_idx] = token_idxs
            all_scores[batch_idx] = token_scores

        return all_tokens, all_scores

