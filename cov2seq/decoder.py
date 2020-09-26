import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 embeddings,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layer,
                 kernel_size,
                 dropout,
                 trg_pad_idx,
                 max_length=100):
        super(Decoder, self).__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).cuda()

        # self.tok_embedding = nn.Embedding(output_dim, emb_dim, _weight=embeddings.float())
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hid_dim,
                      out_channels=2*hid_dim,
                      kernel_size=kernel_size)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedding: [batch_size, trg_len, emb_dim]
        # conved:    [batch_size, hid_dim, trg_len]
        # encoder_conved:   [batch_size, src_len, emb_dim]
        # encoder_combined: [batch_size, src_len, emb_dim]

        # conved: [batch_size, trg_len, hid_dim]
        conved = conved.permute(0, 2, 1)

        # conved_emb: [batch_size, trg_len, emb_dim]
        conved_emb = self.attn_hid2emb(conved)

        # conbined: [batch_size, trg_len, emb_dim]
        combined = (conved_emb + embedded) * self.scale

        # encoder_conved: [batch_size, emb_dim, src_len]
        encoder_conved = encoder_conved.permute(0, 2, 1)

        # energy: [batch_size, trg_len, src_len]
        energy = torch.matmul(combined, encoder_conved)

        # attention: [batch_size, trg_len, src_len]
        attention = F.softmax(energy, dim=2)

        # attended_encoding: [batch_size, trg_len, emb_dim]
        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding: [batch_size, trg_len, hid_dim]
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding: [batch_size, hid_dim, trg_len]
        attended_encoding = attended_encoding.permute(0, 2, 1)

        # conved: [batch_size, hid_dim, trg_len]
        conved = conved.permute(0, 2, 1)

        # attended_combined: [batch_size, hid_dim, trg_len]
        attended_combined = (conved + attended_encoding) * self.scale

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg:              [batch_size, trg_len]
        # encoder_conved:   [batch_size, src_len, emb_dim]
        # encoder_combined: [batch_size, src_len, emb_dim]

        batch_size = trg.size(0)
        trg_len    = trg.size(1)

        # pos: [0, 1, 2, ..., trg_len-1]
        pos = torch.arange(0, trg_len)

        # pos: [batch_size, trg_len]
        pos = pos.unsqueeze(0).repeat(batch_size, 1).cuda()

        # tok_emb: [batch_size, trg_len, emb_dim]
        # pos_emb: [batch_size, trg_len, emb_dim]
        tok_emb = self.tok_embedding(trg)
        pos_emb = self.pos_embedding(pos)

        # emb: [batch_size, trg_len, emb_dim]
        emb = tok_emb + pos_emb
        emb = self.dropout(emb)

        # conv_input: [batch_size, trg_len, hid_dim]
        conv_input = self.emb2hid(emb)

        # conv_input: [batch_size, hid_dim, trg_len]
        conv_input = conv_input.permute(0, 2, 1)
        conved = None
        attention = None

        batch_size = conv_input.size(0)
        hid_dim = conv_input.size(1)

        for i, conv in enumerate(self.convs):
            # conv_input: [batch_size, hid_dim, trg_len]
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size-1).fill_(self.trg_pad_idx).cuda()

            # padded_conv_input: [batch_size, hid_dim, trg_len + kernel_size - 1]
            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # conved: [batch_size, 2*hid_dim, trg_len]
            conved = conv(padded_conv_input)

            # conved: [batch_size, hid_dim, trg_len]
            conved = F.glu(conved, dim=1)

            # attention: [batch_size, trg_len, src_len]
            # conved:    [batch_size, hid_dim, trg_len]
            attention, conved = self.calculate_attention(embedded=emb,
                                                         conved=conved,
                                                         encoder_conved=encoder_conved,
                                                         encoder_combined=encoder_combined)

            # conved: [batch_size, hid_dim, trg_len]
            conved = (conved + conv_input) * self.scale

            conv_input = conved

        # conved: [batch_size, trg_len, hid_dim]
        conved = conved.permute(0, 2, 1)

        # conved: [batch_size, trg_len, emb_dim]
        conved = self.hid2emb(conved)

        # output: [batch_size, trg_len, emb_dim]
        output = self.dropout(conved)
        # output: [batch_size, trg_len, output_dim]
        output = self.fc_out(output)

        return output, attention
