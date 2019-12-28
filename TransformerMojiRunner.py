import os
import sys

module_path = os.path.abspath('utils')
sys.path.insert(0, module_path)
sys.path.append("../../")

import torch
from torch import nn
from utils.log import Log
from utils.ModelRunner import ModelRunner
from model.transformerMoji import DeepMoji as moji
# from preprocess.make_vocab import Lang as dataset
from preprocess.pre_processing import origin_dataset as dataset
from preprocess.transfer_vocab import transfer_vocab
import torch

from tqdm import tqdm


class BaseLineRunner(ModelRunner):
    model: nn.Module

    def __init__(self, model_name='test', model=None, device=torch.device('cuda:0')):
        super().__init__(model_name, model, device)
        self.model_name = model_name
        self.model_folder = './save/%s/' % model_name
        self.model = model
        self.optimizer = None
        self.logger = Log(model_name, model_folder=self.model_folder, overwrite=False)
        self.loss_f = nn.CrossEntropyLoss()

    def evaluate_epoch(self, dataset, epoch, mode, load=False):
        if load:
            self.load_model(epoch)
        self.model.eval()
        pbar = tqdm(dataset, dynamic_ncols=True)
        epoch_loss = 0
        epoch_hit = 0
        epoch_cnt = 0
        for batch in pbar:
            inputs = batch['input'].to(self.device)
            input_lens = batch['input_lens']
            label = batch['label'].to(self.device)
            hope = self.model(inputs, input_lens)
            l = self.loss_f(hope, label)
            # self.loss += l.sum()
            pred = hope.argmax(dim=-1)
            epoch_loss += l.sum().item()
            epoch_hit += (pred == label).sum().item()
            epoch_cnt += inputs.size(0)
            pbar.set_description("%s, epoch-%d, loss: %.4f, acc: %.4f" % (mode, epoch, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt))
        return

    def train_epoch(self, dataset, epoch):
        self.model.train()
        pbar = tqdm(dataset, dynamic_ncols=True)
        epoch_loss = 0
        epoch_hit = 0
        epoch_cnt = 0
        for batch in pbar:
            inputs = batch['input'].to(self.device)
            input_lens = batch['input_lens']
            label = batch['label'].to(self.device)
            # print(inputs.shape)
            # print(label)
            hope = self.model(inputs, input_lens)
            l = self.loss_f(hope, label)
            self.loss += l.sum()
            self.optimize(5)
            pred = hope.argmax(dim=-1)
            epoch_loss += l.sum().item()
            epoch_hit += (pred == label).sum().item()
            epoch_cnt += inputs.size(0)
            pbar.set_description("train, epoch-%d, loss: %.4f, acc: %.4f" % (epoch, epoch_loss/epoch_cnt, epoch_hit/epoch_cnt))
        self.save_model(epoch, "%.4f"%(epoch_loss/epoch_cnt))
        return


if __name__ == '__main__':
    pretrain = True

    device = torch.device('cuda:0')

    # lang = dataset('data1_170000', device)
    # # lang = transfer_vocab('SS-Youtube', device)
    # # lang.load_origin_vocab()
    # # lang = Lang('PsychExp', device)
    # # lang.read_raw()
    # train, dev, test, word2index = lang.load_preprocessed(64)
    #
    # # runner = BaseLineRunner('deepmoji', model=moji(len(word2index), load_emb=True, emb_fixed=False, dim=400),
    # # device=device)
    # runner = BaseLineRunner('deepmoji_pretrain', model=moji(len(word2index), load_emb=True, emb_fixed=False, dim=400), device=device)
    # runner.load_model(epoch=3, load_path='./save/deepmoji_pretrain')
    #
    # # # runner = BaseLineRunner('baseline', model=BaseLine(len(word2index), 'SS-Youtube', True, False), device=device)
    # # runner = BaseLineRunner('baseline', model=BaseLine(len(word2index), 'PsychExp', True, False, 7), device=device)
    #
    # runner.set_optimizer(1e-4)
    # # runner.save_model(0, 0)
    # runner.evaluate_epoch(dev, 0, 'dev')
    # runner.evaluate_epoch(test, 0, 'test')
    # for epoch in range(20):
    #     train, dev, test, _ = lang.load_preprocessed(128)
    #     runner.train_epoch(train, epoch)
    #     runner.evaluate_epoch(dev, epoch, 'dev')
    #     runner.evaluate_epoch(test, epoch, 'test')


    if pretrain:
        lang = dataset('data1_170000', device)
        lang.read_raw_data()
        train, dev, test, _ = lang.load_preprocessed(64)
        runner = BaseLineRunner('deepmoji_pretrain', model=moji(len(lang.word2index), load_emb=False, emb_fixed=False, dim=256, classes=1791), device=device)
        runner.load_model(epoch=0, load_path='./save/deepmoji_pretrain')
    else:
        lang = transfer_vocab('SS-Youtube', device)
        # lang = Lang('PsychExp', device)
        lang.load_origin_vocab()
        train, dev, test = lang.load_from_preprocessed(64)

        runner = BaseLineRunner('deepmoji_transfer', model=moji(len(lang.word2index), load_emb=False, emb_fixed=False, dim=256, classes=1791), device=device)
        # runner.load_model(epoch=27, load_path='./save/deepmoji_transfer')
        runner.load_model(epoch=16, load_path='./save/deepmoji_pretrain')
        runner.model.classifier = nn.Linear(9*400, 2)
        runner.model.classifier.weight.data.normal_(0, 0.1)
        runner.model.classifier.bias.data.normal_(0, 0.1)

    runner.set_optimizer(1e-0)
    # runner.save_model(0, 0)
    # runner.evaluate_epoch(dev, 0, 'dev')
    runner.evaluate_epoch(test, 0, 'test')
    for epoch in range(100):
        if pretrain:
            train, dev, test, _ = lang.load_preprocessed(64)
        else:
            train, dev, test = lang.load_from_preprocessed(64)
        runner.train_epoch(train, epoch)
        # runner.evaluate_epoch(dev, epoch, 'dev')
        runner.evaluate_epoch(test, epoch, 'test')
