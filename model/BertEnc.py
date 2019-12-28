import torch
from torch import nn

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# from model.Encoder import Encoder
# from preprocess.make_vocab import Lang
from model.Attention import Attn

import os
import json
from utils.config import Config


class BertEnc(nn.Module):
    def __init__(self, vocab_size, dataset='SS-Youtube', load_emb=True, emb_fixed=False, classes=2):
        super(BertEnc, self).__init__()
        self.dataset = dataset

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.layer_attention = Attn(768, 768)
        self.classifier = nn.Linear(768, classes)

        self.Config = Config()

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

    def bert_parallel(self, input_texts):
        indexed_tokens = []
        attention_masks = []
        for text in input_texts:
            tokenized_text = self.tokenizer.tokenize('[CLS] ' + text + ' [SEP]')
            indexed_tokens.append(self.tokenizer.convert_tokens_to_ids(tokenized_text))
        seq_len = [len(sentence) for sentence in indexed_tokens]
        max_len = max(seq_len)
        for i, tokens in enumerate(indexed_tokens):
            padding = max_len - len(tokens)
            attention_masks.append([1] * len(tokens) + [0] * padding)
            tokens += [0] * padding
        indexed_tokens_tensor = torch.tensor(indexed_tokens).to(self.Config.device)
        attention_masks_tensor = torch.tensor(attention_masks).to(self.Config.device)
        assert indexed_tokens_tensor.shape == attention_masks_tensor.shape
        assert indexed_tokens_tensor.size(0) == attention_masks_tensor.size(0) == len(input_texts)
        assert indexed_tokens_tensor.size(1) == attention_masks_tensor.size(1) == max_len
        enc_states, hid = self.bert.forward(indexed_tokens_tensor, attention_mask=attention_masks_tensor)  # hid: [bsz, len, 768]
        # print('enc: ', enc_states[-1].shape)  # [bsz, len, 768]
        # print('hid: ', hid.shape)  # [bsz, 768]
        feature = enc_states[-1][:, 0, :]  # [bsz, 768]
        return feature, seq_len, enc_states

    def forward(self, input_texts: list):
        # with torch.no_grad():
        #     hids = [self.process("[CLS] %s [SEP]" % text) for text in input_texts]
        # hids = [self.process("[CLS] %s [SEP]" % text) for text in input_texts]
        feature, _, _ = self.bert_parallel(input_texts)
        assert feature.size(0) == len(input_texts) and feature.size(1) == 768
        hope = self.classifier(feature)
        return hope


if __name__ == '__main__':
    pass
