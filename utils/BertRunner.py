import os
import sys

module_path = os.path.abspath('.')
sys.path.insert(0, module_path)
sys.path.append("../../")

import torch
from torch import nn
from utils.log import Log
from utils.ModelRunner import ModelRunner
from model.Attention import Attn

from tqdm import tqdm


class BertRunner(ModelRunner):
    model: nn.Module

    def __init__(self, model_name='test', model=nn.Module, device=torch.device('cuda:0')):
        super().__init__(model_name, model, device)
        self.model_name = model_name
        self.model_folder = './save/%s/' % model_name
        self.model = model.to(self.device)
        self.optimizer = None
        self.logger = Log(model_name, model_folder=self.model_folder, overwrite=False)
        self.loss_f = nn.CrossEntropyLoss()
        self.best_acc = 0

    def evaluate_epoch(self, dataset, epoch, mode, load=False):
        if load:
            self.load_model(epoch)
        self.model.eval()
        pbar = tqdm(dataset, dynamic_ncols=True)
        epoch_loss = 0
        epoch_hit = 0
        epoch_cnt = 0
        for batch in pbar:
            label = batch['label'].to(self.device)
            input_texts = batch['text']
            hope = self.model.forward(input_texts)
            l = self.loss_f(hope, label)
            pred = hope.argmax(dim=-1)
            epoch_loss += l.sum().item()
            epoch_hit += (pred == label).sum().item()
            epoch_cnt += hope.size(0)
            pbar.set_description(
                "%s, epoch-%d, loss: %.4f, acc: %.4f" % (mode, epoch, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt))
        self.logger.write(
            "epoch-%d %s loss: %.4f acc: %.4f" % (epoch, mode, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt))
        if self.best_acc < epoch_loss / epoch_cnt and mode == 'test':
            self.best_acc = epoch_loss / epoch_cnt
            if self.best_acc > 0.85:
                self.save_best(epoch, epoch_loss / epoch_cnt)
        return epoch_loss / epoch_cnt

    def train_epoch(self, dataset, epoch):
        self.model.train()
        pbar = tqdm(dataset, dynamic_ncols=True)
        epoch_loss = 0
        epoch_hit = 0
        epoch_cnt = 0
        for batch in pbar:
            label = batch['label'].to(self.device)
            input_texts = batch['text']
            hope = self.model.forward(input_texts)
            l = self.loss_f(hope, label)
            self.loss += l.sum()
            self.optimize(5)
            pred = hope.argmax(dim=-1)
            epoch_loss += l.sum().item()
            epoch_hit += (pred == label).sum().item()
            epoch_cnt += hope.size(0)
            pbar.set_description(
                "train, epoch-%d, loss: %.4f, acc: %.4f" % (epoch, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt))
        # self.save_model(epoch, "%.4f" % (epoch_loss / epoch_cnt))
        return epoch_loss / epoch_cnt