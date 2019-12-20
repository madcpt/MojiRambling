from torch import nn
import torch
import json
import os
import numpy as np


class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        # self.encoder = nn.GRU(input_size=400, hidden_size=400, num_layers=2, batch_first=True, bidirectional=True)
        self.encoder = nn.GRU(input_size=400, hidden_size=400, num_layers=8, batch_first=True)
        self.dropout_layer = nn.Dropout(0.1)

    def forward(self, input_embedding, input_lens, hidden=None):
        input_lens = np.array(input_lens)
        sort_idx = np.argsort(-input_lens)
        input_lengths = input_lens[sort_idx]
        sort_input_seq = input_embedding[sort_idx]
        embedded = self.dropout_layer(sort_input_seq)   # 现在 embedding 层统一了，就放在主模型
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, en_hidden = self.encoder(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        invert_sort_idx = np.argsort(sort_idx)
        en_hidden = en_hidden.transpose(0, 1)
        outputs = outputs[invert_sort_idx]
        en_hidden = en_hidden[invert_sort_idx].transpose(0, 1)
        return outputs, en_hidden


if __name__ == '__main__':
    pass
