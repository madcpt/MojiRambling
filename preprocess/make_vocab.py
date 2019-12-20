import pickle
import numpy
import random
from tqdm import tqdm
import json
import os

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from embeddings import GloveEmbedding, KazumaCharEmbedding


class Lang(object):
    def __init__(self, dataset, device=None):
        self.device = device  # deprecated
        self.PAD = 1
        self.UNK = 0
        self.word2index = {'UNK': self.UNK, 'PAD': self.PAD}
        self.dataset = dataset
        self.dataset_path = './data/%s/' % dataset

    def load_raw_data(self):
        with open(self.dataset_path + 'raw.pickle', 'rb') as f:
            content = pickle.load(f)
        raw = content['texts']
        label = content['info']
        data = [{'text': raw[i], 'label': label[i]['label']} for i, _ in enumerate(label)]
        max_len = 0
        for i, sentences in enumerate(raw):
            tokens = sentences.split(' ')
            max_len = max(max_len, len(tokens))
            for token in tokens:
                if token not in self.word2index.keys():
                    self.word2index[token] = len(self.word2index)
            data[i]['input'] = [self.word2index[token] for token in tokens]
            data[i]['input_lens'] = len(data[i]['input'])
        for sample in data:
            sample['input'] = sample['input'] + ([self.PAD] * (max_len - len(sample['input'])))
        train, dev, test = self.split_dataset(data, 800, 100, 1242)
        self.dump_preprocessed(train, dev, test, self.word2index)
        index2word = {v:k for k,v in self.word2index.items()}
        pretrained_emb = self.dataset_path + 'emb{}.json'.format(str(len(index2word)))
        if not os.path.exists(pretrained_emb):
            self.dump_pretrained_emb(self.word2index, index2word, pretrained_emb)

    @staticmethod
    def split_dataset(data, train_num=0, dev_num=0, test_num=0):
        try:
            assert len(data) == train_num + dev_num + test_num
        except AssertionError:
            train_num = int(0.8 * len(data))
            test_num = int(0.1 * len(data))
            dev_num = len(data) - train_num - test_num
        order = list(range(len(data)))
        random.shuffle(order)
        train_split = order[:train_num]
        dev_split = order[train_num: train_num + dev_num]
        test_split = order[train_num + dev_num: train_num + dev_num + test_num]
        train = [data[i] for i in train_split]
        dev = [data[i] for i in dev_split]
        test = [data[i] for i in test_split]
        return train, dev, test

    def dump_preprocessed(self, train, dev, test, word2index):
        dump_object = {'train': train, 'dev': dev, 'test': test, 'word2index': word2index}
        with open(self.dataset_path + 'preprocessed.pickle', 'wb') as f:
            pickle.dump(dump_object, f)

    def load_preprocessed(self, batch_size=4):
        with open(self.dataset_path + 'preprocessed.pickle', 'rb') as f:
            preprocessed = pickle.load(f)
        self.word2index = preprocessed['word2index']

        def make_dataloader(data):
            for sample in data:
                sample['input'] = torch.tensor(sample['input'])
            dataset = DataLoader(data, batch_size, shuffle=True)
            return dataset

        train = make_dataloader(preprocessed['train'])
        dev = make_dataloader(preprocessed['dev'])
        test = make_dataloader(preprocessed['test'])
        return train, dev, test, self.word2index

    # dump_path: data/emb{}.json
    @staticmethod
    def dump_pretrained_emb(word2index, index2word, dump_path):
        print("Dumping pretrained embeddings...")
        embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
        E = []
        for i in tqdm(range(len(word2index.keys()))):
            w = index2word[i]
            e = []
            for emb in embeddings:
                e += emb.emb(w, default='zero')
            E.append(e)
        with open(dump_path, 'wt') as f:
            json.dump(E, f)


if __name__ == '__main__':
    lang = Lang('SS-Youtube', None)
    lang.load_raw_data()
    # train, dev, test, word2index = lang.load_preprocessed()
