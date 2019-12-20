from torch import nn
import torch


class Attn(nn.Module):
    def __init__(self, q_hidden_size, k_hidden_size, mode='general'):
        # encoder_outputs: (batch, len, q_hidden_size)
        # hidden: (batch, k_hidden)
        super(Attn, self).__init__()
        self.modes = ['dot', 'general', 'concat']
        assert mode in self.modes
        self.mode = mode

        if mode is 'general':
            self.att = nn.Linear(q_hidden_size, k_hidden_size)

    def _dot_socre(self, encoder_outputs, hidden):
        score = torch.sum(hidden * encoder_outputs, dim=2)
        return score

    def _general_score(self, encoder_outputs, hidden):
        # (batch, len, dim) -> (batch, len, k_hidden_size)
        energy = self.att(encoder_outputs)
        # (batch, k_hidden) -> (batch, len, k_hidden)
        hidden = torch.repeat_interleave(hidden.unsqueeze(1), encoder_outputs.size(1), dim=1)
        score = torch.sum(hidden * energy, dim=2)
        return score

    def forward(self, encoder_outputs, hidden):
        # encoder_outputs: (batch, len, q_hidden_size)
        # hidden: (batch, k_hidden)
        att_score = None
        if self.mode is 'dot':
            att_score = self._dot_socre(encoder_outputs, hidden)
        elif self.mode is 'general':
            att_score = self._general_score(encoder_outputs, hidden)
        # return th.softmax(att_score, dim=1).unsqueeze(1)
        return att_score
