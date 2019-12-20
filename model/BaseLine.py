import torch
from torch import nn
from model.Encoder import Encoder
import os
import json


class BaseLine(nn.Module):
    def __init__(self, vocab_size, dataset='SS-Youtube', load_emb=True, emb_fixed=False, classes=2):
        super(BaseLine, self).__init__()
        self.dataset = dataset
        self.encoder = Encoder(vocab_size)
        self.word_embedding = nn.Embedding(vocab_size, 400)
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

        # self.classifier = nn.Sequential(nn.Linear(800, 200), nn.Tanh(), nn.Linear(200, 2))
        self.classifier = nn.Linear(800, classes)

    def forward(self, inputs, input_lens):
        input_embedding = self.word_embedding(inputs)
        output, hidden = self.encoder(input_embedding, input_lens)
        # output = torch.cat([output[-1], output[-2]], dim=-1)
        hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        hope = self.classifier(hidden)
        return hope


if __name__ == '__main__':
    pass
