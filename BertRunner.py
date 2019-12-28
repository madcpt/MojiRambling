import os
import sys

module_path = os.path.abspath('utils')
sys.path.insert(0, module_path)
sys.path.append("../../")

import torch
from torch import nn
from utils.log import Log
from utils.config import config
from utils.ModelRunner import ModelRunner
from model.Attention import Attn
from model.BertEnc import BertEnc
from preprocess.make_vocab import Lang
import torch


from tqdm import tqdm


class BertRunner(ModelRunner):
    def __init__(self, model_name: 'test', model: nn.Module):
        super().__init__(model_name, model, config.device)
        self.model_name = model_name
        self.model_folder = './save/%s/' % model_name
        self.model = model.to(config.device)
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
        tp, tn, fp, fn = 0.01, 0.01, 0.01, 0.01
        for batch in pbar:
            label = batch['label'].to(self.device).long()
            input_texts = batch['text']
            hope = self.model.forward(input_texts)
            l = self.loss_f(hope, label)
            pred = hope.argmax(dim=-1)
            epoch_loss += l.sum().item()
            epoch_hit += (pred == label).sum().item()
            epoch_cnt += hope.size(0)
            tp += ((pred == label) & (pred == 1)).sum().item()
            tn += ((pred == label) & (pred != 1)).sum().item()
            fp += ((pred != label) & (pred == 1)).sum().item()
            fn += ((pred != label) & (pred != 1)).sum().item()
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            pbar.set_description(
                "%s, epoch-%d, loss: %.4f, acc: %.4f, P: %.4f, R: %.4f, F1: %.4f" % (mode, epoch, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt, precision, recall, f1))
        self.logger.write(
            "epoch-%d %s loss: %.4f acc: %.4f, P: %.4f, R: %.4f, F1: %.4f" % (epoch, mode, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt, precision, recall, f1))
        if self.best_acc < epoch_loss / epoch_cnt and mode == 'test':
            self.best_acc = epoch_loss / epoch_cnt
            if self.best_acc > 0.85 and config.save_model:
                self.save_best(epoch, epoch_loss / epoch_cnt)
        return epoch_loss / epoch_cnt

    def train_epoch(self, dataset, epoch):
        self.model.train()
        pbar = tqdm(dataset, dynamic_ncols=True)
        epoch_loss = 0
        epoch_hit = 0
        epoch_cnt = 0
        for batch in pbar:
            label = batch['label'].to(self.device).long()
            input_texts = batch['text']
            hope = self.model.forward(input_texts)
            l = self.loss_f(hope, label)
            self.loss += l.sum()
            self.optimize(20)
            pred = hope.argmax(dim=-1)
            epoch_loss += l.sum().item()
            epoch_hit += (pred == label).sum().item()
            epoch_cnt += hope.size(0)
            pbar.set_description(
                "train, epoch-%d, loss: %.4f, acc: %.4f" % (epoch, epoch_loss / epoch_cnt, epoch_hit / epoch_cnt))
        # self.save_model(epoch, "%.4f" % (epoch_loss / epoch_cnt))
        return epoch_loss / epoch_cnt


if __name__ == '__main__':
    device = config.device
    pretrain = config.mode == 'pretrain'
    training = config.train

    lang = Lang('SS-Youtube', config.device)
    # lang = Lang('SS-Twitter', config.device)
    # lang = Lang('PsychExp', config.device)
    lang.load_preprocessed(64)
    train, dev, test, word2index = lang.load_preprocessed(batch_size=8)

    runner = BertRunner('bert', model=BertEnc(len(word2index), 'SS-Youtube', True, False, 2))

    if training:
        runner.set_optimizer(1e-7)
        # runner.save_model(0, 0)
        # runner.evaluate_epoch(dev, 0, 'dev')
        # runner.evaluate_epoch(test, 0, 'test')
        for epoch in range(100):
            train, dev, test, _ = lang.load_preprocessed(16)
            runner.train_epoch(train, epoch)
            runner.evaluate_epoch(dev, epoch, 'dev')
            runner.evaluate_epoch(test, epoch, 'test')
    else:
        # runner.evaluate_epoch(test, 0, 'test')
        raise NotImplementedError
