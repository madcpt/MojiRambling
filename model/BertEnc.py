import torch
from torch import nn

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from model.Encoder import Encoder
# from preprocess.make_vocab import Lang
from model.Attention import Attn

import os
import json


class BertEnc(nn.Module):
    def __init__(self, vocab_size, dataset='SS-Youtube', load_emb=True, emb_fixed=False, classes=2):
        super(BertEnc, self).__init__()
        self.dataset = dataset

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.layer_attention = Attn(768, 768)
        self.classifier = nn.Linear(768*2, classes)

    def process_(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        _, hid = self.bert(tokens_tensor, segments_tensors)
        return hid

    def process(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        enc, hid = self.bert(tokens_tensor, segments_tensors)
        feature = enc[-1][:, 0, :]
        c = torch.cat([feature, hid], dim=-1)
        return c

    def forward(self, input_texts: list):
        with torch.no_grad():
            hids = [self.process("[CLS] %s [SEP]" % text) for text in input_texts]
        hids = torch.cat(hids, dim=0)
        assert hids.size(0) == len(input_texts) and hids.size(1) == 768*2
        hope = self.classifier(hids)
        return hope


if __name__ == '__main__':
    pass
