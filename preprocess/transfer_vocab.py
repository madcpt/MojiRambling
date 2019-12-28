# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:30:18 2019

@author: SJTUwwz
"""

import os
import sys
import pickle

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

from preprocess.make_vocab import Lang


class transfer_vocab(Lang):
    def __init__(self, dataset, device=None):
        super().__init__(dataset, device)

    def load_origin_vocab(self, vocab_path='./data/data1_170000/preprocessed.pickle'):
        with open(vocab_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            content = u.load()
        self.word2index = content['word2index']
        # print(content['dev'][0])

    def load_from_preprocessed(self, batch_size=8):
        with open(self.dataset_path + 'preprocessed.pickle', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            content = u.load()

        def translate_vocab(data):
            transfered_data = []
            for i, sample in enumerate(data):
                # print(sample)
                tokenized = sample['text'].split(' ')
                transfered_index = [self.word2index[token] if token in self.word2index.keys() else self.word2index['UNK'] for token in tokenized]
                transfered_index += (len(sample['input']) - sample['input_lens']) * [self.word2index['PAD']]
                assert len(transfered_index) == sample['input'].__len__()
                transfered_data.append({'text': sample['text'], 'label': sample['label'], 'input': transfered_index, 'input_lens': sample['input_lens']})
            return transfered_data

        train = translate_vocab(content['train'])
        dev = translate_vocab(content['dev'])
        test = translate_vocab(content['test'])
        train_loader = self.make_dataloader(train, batch_size, True)
        dev_loader = self.make_dataloader(dev, batch_size, False)
        test_loader = self.make_dataloader(test, batch_size, False)

        return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    test_dataset = transfer_vocab("SS-Youtube")  # you can set the size of dataset with beign and end
    test_dataset.load_origin_vocab()
    train_loader, dev_loader, test_loader = test_dataset.load_from_preprocessed()
    for s in train_loader:
        print(s)
        break
