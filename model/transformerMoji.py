import torch
from torch import nn
# from model.Encoder import Encoder

import os
import json
import numpy as np

from model.attlayer import Attention
import math


# class Attn(nn.Module):
#     def __init__(self, q_hidden_size, k_hidden_size, mode='general'):
#         super(Attn, self).__init__()
#         self.modes = ['dot', 'general', 'concat']
#         assert mode in self.modes
#         self.mode = mode
#
#         if mode is 'general':
#             self.att = nn.Linear(q_hidden_size, k_hidden_size)
#
#     def _dot_socre(self, encoder_outputs, hidden):
#         score = torch.sum(hidden * encoder_outputs, dim=2)
#         return score
#
#     def _general_score(self, encoder_outputs, hidden):
#         # (batch, len, dim) -> (batch, len, k_hidden_size)
#         energy = self.att(encoder_outputs)
#         # (batch, k_hidden) -> (batch, len, k_hidden)
#         hidden = torch.repeat_interleave(hidden.unsqueeze(1), encoder_outputs.size(1), dim=1)
#         score = torch.sum(hidden * energy, dim=2)
#         return score
#
#     def forward(self, encoder_outputs, hidden):
#         # encoder_outputs: (batch, len, q_hidden_size)
#         # hidden: (batch, k_hidden)
#         att_score = None
#         if self.mode is 'dot':
#             att_score = self._dot_socre(encoder_outputs, hidden)
#         elif self.mode is 'general':
#             att_score = self._general_score(encoder_outputs, hidden)
#         # return th.softmax(att_score, dim=1).unsqueeze(1)
#         return att_score

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


def generate_src_mask(input_embedding, input_lens):
    device = input_embedding.device
    batch_size = input_embedding.size(0)
    mask = (torch.zeros(size=(batch_size, input_embedding.size(1))) == 0).to(device)
    for batch_num in range(batch_size):
        mask[batch_num, :input_lens[batch_num]] = False
    return mask.bool()


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim, rnn_hidden):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.rnn_hidden = rnn_hidden
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input_embedding, input_lens, src_mask, hidden=None):
        input_embedding = input_embedding.transpose(1, 0).contiguous()
        outputs = self.encoder(input_embedding, src_key_padding_mask=src_mask)
        return outputs, None


class DeepMoji(nn.Module):
    def __init__(self, vocab_size, dataset='data1_170000', load_emb=True, emb_fixed=False, dim=256, classes=1791):
        super(DeepMoji, self).__init__()
        self.dataset = dataset
        self.dropout_layer = nn.Dropout(0.1)
        self.dim = dim
        self.rnn_hidden = dim
        self.positional_embedding = PositionalEncoder(self.dim, 100)
        self.encoder1 = Encoder(vocab_size, dim=dim, rnn_hidden=self.rnn_hidden)
        self.encoder2 = Encoder(vocab_size, dim=self.rnn_hidden, rnn_hidden=self.rnn_hidden)
        self.word_embedding = nn.Embedding(vocab_size, self.dim)
        if load_emb:
            self.word_embedding.weight.data.normal_(0, 0.1)
            with open(os.path.join("data/%s/" % dataset, 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.word_embedding.weight.data.new
            self.word_embedding.weight.data.copy_(new(E))
            self.word_embedding.weight.requires_grad = True
            if not emb_fixed:
                print("Encoder embedding requires_grad ", self.word_embedding.weight.requires_grad)
        if emb_fixed:
            self.word_embedding.weight.requires_grad = False
            print("Encoder embedding requires_grad ", self.word_embedding.weight.requires_grad)

        self.attn = Attention(3 * self.dim)

        self.classfier = nn.Linear(self.dim * 3, classes)

    def forward(self, inputs, input_lens):
        inputs = inputs[:, :max(input_lens)]
        input_embedding = self.word_embedding(inputs)
        # input_embedding = self.positional_embedding(input_embedding)
        src_mask = generate_src_mask(input_embedding, input_lens)
        # print(input_embedding.shape)
        output1, _ = self.encoder1(input_embedding, input_lens, src_mask)  # hidden: [n*bi, bsz, hidden]
        output1 = output1.transpose(1, 0).contiguous()
        # print(output1.shape)
        output2, _ = self.encoder2(output1, input_lens, src_mask)  # hidden: [n*bi, bsz, hidden]
        output2 = output2.transpose(1, 0).contiguous()
        # print(output2.shape)
        # output = torch.cat([output[-1], output[-2]], dim=-1)
        attn_input = torch.cat([output1, output2, input_embedding], dim=-1)  # [bsz, 9 * 256]
        attned_sum, atten_score = self.attn(attn_input, input_lens)
        hope = self.classfier(attned_sum)
        # print(hope.shape)
        return hope


if __name__ == '__main__':
    pass
