import torch
from torch import nn
# from model.Encoder import Encoder

import os
import json
import numpy as np

from model.attlayer import Attention


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


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim, rnn_hidden):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.rnn_hidden = rnn_hidden
        self.encoder = nn.GRU(input_size=self.dim, hidden_size=self.rnn_hidden, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input_embedding, input_lens, hidden=None):
        input_lens = np.array(input_lens)
        sort_idx = np.argsort(-input_lens)
        input_lengths = input_lens[sort_idx]
        sort_input_seq = input_embedding[sort_idx]
        embedded = sort_input_seq
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, en_hidden = self.encoder(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        invert_sort_idx = np.argsort(sort_idx)
        en_hidden = en_hidden.transpose(0, 1)
        outputs = outputs[invert_sort_idx]
        en_hidden = en_hidden[invert_sort_idx].transpose(0, 1)  # [n*bi , bsz, 400*bi]
        return outputs, en_hidden


class DeepMoji(nn.Module):
    def __init__(self, vocab_size, dataset='data1_170000', load_emb=True, emb_fixed=False, dim=256):
        super(DeepMoji, self).__init__()
        self.dataset = dataset
        self.dropout_layer = nn.Dropout(0.1)
        self.dim = dim
        self.rnn_hidden = 2 * dim
        self.encoder1 = Encoder(vocab_size, dim=dim, rnn_hidden=self.rnn_hidden)
        self.encoder2 = Encoder(vocab_size, dim=2*self.rnn_hidden, rnn_hidden=self.rnn_hidden)
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

        self.attn = Attention(9 * self.dim)

        self.classfier = nn.Linear(self.dim*9, 1791)

    def forward(self, inputs, input_lens):
        inputs = inputs[:, :max(input_lens)]
        input_embedding = self.word_embedding(inputs)
        output1, hidden = self.encoder1(input_embedding, input_lens)  # hidden: [n*bi, bsz, hidden]
        output2, hidden = self.encoder2(output1, input_lens)  # hidden: [n*bi, bsz, hidden]
        # output = torch.cat([output[-1], output[-2]], dim=-1)
        attn_input = torch.cat([output1, output2, input_embedding], dim=-1)  # [bsz, 9 * 256]
        attned_sum, atten_score = self.attn(attn_input, input_lens)
        hope = self.classfier(attned_sum)
        # print(hope.shape)
        return hope


if __name__ == '__main__':
    pass
