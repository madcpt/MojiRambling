import os
import sys

module_path = os.path.abspath('utils')
sys.path.insert(0, module_path)
sys.path.append("../../")

import torch
from torch import nn
from utils.log import Log
# from model.ToyModel import ToyModel
from utils.ModelRunner import ModelRunner
# from model.BaseLine import BaseLine as moji
from model.deepmoji import DeepMoji as moji
# from preprocess.make_vocab import Lang as dataset
from preprocess.pre_processing import Lang as lang
from preprocess.pre_processing import origin_dataset as dataset
from preprocess.transfer_vocab import transfer_vocab
import torch
from utils.config import config

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
    device = config.device
    pretrain = config.mode == 'pretrain'
    training = config.train

    if pretrain:
        lang = dataset('data1_170000', device)
        lang.read_raw_data()
        train, _, test, word2index = lang.load_preprocessed(64)
        runner = BaseLineRunner('glovemoji', model=moji(len(word2index), load_emb=True, emb_fixed=False, dim=400, classes=1791), device=device)
    else:
        lang = transfer_vocab('SS-Youtube', device)
        # lang = Lang('PsychExp', device)
        lang.load_origin_vocab()
        train, _, test = lang.load_from_preprocessed(64)
        runner = BaseLineRunner('glovemoji_transfer', model=moji(len(lang.word2index), load_emb=False, emb_fixed=False, dim=400, classes=2), device=device)
        if training:
            runner.load_model(epoch=16, load_path='./save/glovemoji_pretrain')
            runner.model.classifier = nn.Linear(9*400, 2)
            runner.model.classifier.weight.data.normal_(0, 0.1)
            runner.model.classifier.bias.data.normal_(0, 0.1)
        else:
            runner.load_model(epoch=6, load_path='./save/glovemoji_transfer')

    if training:
        runner.set_optimizer(1e-0)
        # runner.save_model(0, 0)
        for epoch in range(100):
            train, _, test = lang.load_from_preprocessed(128)
            runner.train_epoch(train, epoch)
            runner.train_epoch(test, epoch)
            runner.evaluate_epoch(test, epoch, 'test')
    else:
        runner.evaluate_epoch(test, 0, 'test')
