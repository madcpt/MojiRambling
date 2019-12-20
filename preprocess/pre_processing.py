# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:30:18 2019

@author: SJTUwwz
"""

import os
import sys

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

from preprocess.make_vocab import Lang


class origin_dataset(Lang):
    def __init__(self, dataset, device=None):
        super().__init__(dataset, device)

    def read_raw_data(self):
        sentences = []
        labels = []  # 1790
        f = open(self.dataset_path + 'dataset.txt', 'r', encoding='utf-8')
        lines = f.readlines()
        max_len = 0
        for line in lines:
            line = line.rstrip()
            split_line = line.split(" : ")
            sentences.append(split_line[0])
            labels.append(int(split_line[1]))
            words = split_line[0].split(" ")
            max_len = max(len(words), max_len)
            for word in words:
                if word not in self.word2index.keys():
                    self.word2index[word] = len(self.word2index)
        assert len(sentences) == len(labels)
        data = [{'text': sentences[i], 'label': labels[i]} for i, _ in enumerate(labels)]
        for i, sample in enumerate(data):
            sample['input'] = [self.word2index[word] for word in sample['text'].split(' ')]
            sample['input_lens'] = len(sample['input'])
            sample['input'] += [self.PAD] * (max_len - sample['input_lens'])
        train, dev, test = self.split_dataset(data)
        self.dump_preprocessed(train, dev, test, self.word2index)
        index2word = {v: k for k, v in self.word2index.items()}
        pretrained_emb = self.dataset_path + 'emb{}.json'.format(str(len(index2word)))
        if not os.path.exists(pretrained_emb):
            self.dump_pretrained_emb(self.word2index, index2word, pretrained_emb)


if __name__ == "__main__":
    test_dataset = origin_dataset("data1_170000")  # you can set the size of dataset with beign and end
    test_dataset.read_raw_data()
    # print(len(test_dataset.vocabulary))  # vocabulary is with test_set
    # train_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # for se, la in train_loader:
    #     print(se, la)
    #     break
