import os
import sys
import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from seq2seq_attention.attention import Attention


class Decoder(nn.Module):
    def __init__(self, embeddings, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attn_dim, n_layer, dropout):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.n_layer = n_layer
        self.dropout = dropout

        # self.embedding = nn.Embedding(input_dim, emb_dim)
        # self.embedding = nn.Embedding(input_dim, emb_dim, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(embeddings.float(), freeze=False)
        # self.embedding.requires_grad = False

        self.attention = Attention(enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim, attn_dim=attn_dim)
        self.rnn = nn.GRU(input_size=2 * enc_hid_dim + emb_dim,
                          hidden_size=dec_hid_dim,
                          num_layers=n_layer,
                          dropout=dropout,
                          bidirectional=False)
        self.linear = nn.Linear(2 * enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, decoder_hidden, encoder_outputs):
        # decoder_hidden:  [batch_size, dec_hid_dim]
        # encoder_outputs: [seq_len, batch_size, 2*enc_hid_dim]

        # input_ids: [1, batch_size]
        # emb:       [1, batch_size, emb_dim]
        input_ids = input_ids.unsqueeze(0)
        emb = self.embedding(input_ids)
        emb = self.dropout(emb)

        # attention: [batch_size, seq_len]
        attention = self.attention(decoder_hidden, encoder_outputs)

        # attention: [batch_size, 1, seq_len]
        attention = attention.unsqueeze(1)

        # encoder_outputs: [batch_size, seq_len, 2*enc_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # weighted: [batch_size, 1, 2*enc_hid_dim]
        weighted = torch.bmm(attention, encoder_outputs)

        # weighted: [1, batch_size, 2*enc_hid_dim]
        weighted = weighted.permute(1, 0, 2)

        # rnn_input: [1, batch_size, 2*enc_hid_dim+emb_dim]
        rnn_input = torch.cat((emb, weighted), dim=2)

        # decoder_hidden: [1, batch_size, dec_hid_dim]
        decoder_hidden = decoder_hidden.unsqueeze(0)

        # rnn_output:     [1, batch_size, dec_hid_dim]
        # decoder_hidden: [1, batch_size, dec_hid_dim]
        rnn_output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)

        # emb:            [batch_size, emb_dim]
        # rnn_output:     [batch_size, dec_hid_dim]
        # weighted:       [batch_size, 2*enc_hid_dim]
        emb = emb.squeeze(0)
        rnn_output = rnn_output.squeeze(0)
        weighted = weighted.squeeze(0)

        # decoder_hidden: [batch_size, dec_hid_dim]
        decoder_hidden = decoder_hidden.squeeze(0)

        # output: [batch_size, 2*enc_hid_dim+dec_hid_dim+emb_dim]
        output = torch.cat((rnn_output, weighted, emb), dim=1)

        # output: [batch_size, dec_hid_dim]
        # decoder_hidden: [batch_size, dec_hid_dim]
        output = self.linear(output)
        return output, decoder_hidden
