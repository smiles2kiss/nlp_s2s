import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from seq2seq_attention1.attention import Attention


class Decoder(nn.Module):
    def __init__(self, embeddings, output_size, embedding_size, hidden_size, n_layer, dropout):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        dropout = 0 if n_layer == 1 else dropout

        # self.embedding = nn.Embedding(output_dim, emb_dim)
        # self.embedding = nn.Embedding(output_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        self.emb_dropout = nn.Dropout(dropout)
        # self.embedding.requires_grad = False

        # Attention
        self.gru = nn.GRU(input_size=2 * hidden_size + embedding_size,
                          hidden_size=hidden_size,
                          num_layers=n_layer,
                          dropout=dropout,
                          bidirectional=False)
        self.attn = Attention(enc_hid_dim=hidden_size, dec_hid_dim=hidden_size)
        self.linear = nn.Linear(2 * hidden_size + hidden_size + embedding_size, output_size)

    # Attention
    def forward(self, input_ids, decoder_hidden, encoder_outputs):
        # run decoder one step(word) at a time
        # decoder_hidden:  [batch_size, hidden_size]
        # encoder_outputs: [seq_len, batch_size, 2*hidden_size]

        # input_ids: [1, batch_size]
        # emb:       [1, batch_size, hidden_size]
        input_ids = input_ids.unsqueeze(0)
        emb = self.embedding(input_ids)
        emb = self.emb_dropout(emb)

        # attn_weights:    [batch_size, seq_len]
        attn_weights = self.attn(decoder_hidden, encoder_outputs)

        # attn_weights:    [batch_size, 1, seq_len]
        attn_weights = attn_weights.unsqueeze(1)

        # encoder_outputs: [batch_size, seq_len, 2*hidden_size]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # context: [batch_size, 1, 2*hidden_size]
        context = attn_weights.bmm(encoder_outputs)

        # context: [1, batch_size, 2*hidden_size]
        context = context.permute(1, 0, 2)

        # rnn_input: [1, batch_size, 2*hidden_size + embedding_size]
        rnn_input = torch.cat((emb, context), dim=2)

        # decoder_hidden: [1, batch_size, hidden_size]
        decoder_hidden = decoder_hidden.unsqueeze(0)

        # rnn_output:     [1, batch_size, hidden_size]
        # decoder_hidden: [1, batch_size, hidden_size]
        rnn_output, decoder_hidden = self.gru(rnn_input, decoder_hidden)

        # emb:            [batch_size, embedding_size]
        # context:        [batch_size, 2*hidden_size]
        # rnn_output:     [batch_size, hidden_size]
        emb = emb.squeeze(0)
        context = context.squeeze(0)
        rnn_output = rnn_output.squeeze(0)

        # decoder_hidden: [batch_size, hidden_size]
        decoder_hidden = decoder_hidden.squeeze(0)

        # prediction: [batch_size, 2*hidden_size + hidden_size + embedding_size]
        prediction = torch.cat((rnn_output, context, emb), dim=1)

        # prediction: [batch_size, output_dim]
        prediction = self.linear(prediction)
        return prediction, decoder_hidden
