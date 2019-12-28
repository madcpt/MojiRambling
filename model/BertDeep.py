from model.BertEnc import BertEnc
from model.deepmoji import Encoder
from model.attlayer import Attention

from utils.config import config

from torch import nn
import torch


class BertDeep(nn.Module):
    def __init__(self, vocab_size, dataset, classses):
        super().__init__()
        self.bertEncoder = BertEnc(vocab_size, dataset, True, False, classses)
        self.dim = 768
        self.rnn_hidden = 2 * self.dim
        self.encoder1 = Encoder(vocab_size, dim=self.dim, rnn_hidden=self.rnn_hidden)
        self.encoder2 = Encoder(vocab_size, dim=2*self.rnn_hidden, rnn_hidden=self.rnn_hidden)
        self.attn = Attention(9 * self.dim)
        self.classfier = nn.Linear(self.dim * 9, classses)

    def forward(self, input_texts):
        _, seq_len, enc_states = self.bertEncoder.bert_parallel(input_texts)  # enc_state[-1]: [bsz, len, 768]
        seq_len = torch.tensor(seq_len).cpu()
        output1, hidden = self.encoder1(enc_states[-1], seq_len)  # hidden: [n*bi, bsz, hidden]
        output2, hidden = self.encoder2(output1, seq_len)  # hidden: [n*bi, bsz, hidden]
        # output = torch.cat([output[-1], output[-2]], dim=-1)
        attn_input = torch.cat([output1, output2, enc_states[-1]], dim=-1)  # [bsz, 9 * 256]
        attned_sum, atten_score = self.attn(attn_input, seq_len)
        hope = self.classfier(attned_sum)
        # print(hope.shape)
        return hope




