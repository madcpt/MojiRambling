from typing import Dict, Any, Callable, Tuple

import torch
from torch import nn
from utils.log import Log
# from model.ToyModel import ToyModel


class ModelRunner(object):
    model: nn.Module

    def __init__(self, model_name='test', model=nn.Module, device=torch.device('cuda')):
        self.model_name = model_name
        self.model_folder = './save/%s/' % model_name
        self.model = model.to(device)
        self.optimizer = None
        self.logger = Log(model_name, model_folder=self.model_folder, overwrite=False)
        self.loss = 0
        self.device = device

    def set_optimizer(self, optimizer='Adam', lr=1e-4):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save_best(self, epoch=0, acc=0.0):
        save_epoch = "{}ACC-{.4f}.params".format(self.model_folder, acc)
        print("saving to %s" % save_epoch)
        torch.save({'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': 0
                    },
                   save_epoch)

    def load_best(self, acc=0.0, map_location=None):
        save_epoch = "{}/ACC-{.4f}.params".format(self.model_folder, acc)
        print("loading %s" % save_epoch)
        checkpoint = torch.load(save_epoch, map_location=map_location)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("loaded: epoch-%d loss-%.4f" % (checkpoint['epoch'], checkpoint['loss']))

    def save_model(self, epoch=0, loss=0.0):
        save_epoch = "{}epoch-{}.params".format(self.model_folder, str(epoch))
        print("saving to %s" % save_epoch)
        torch.save({'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': loss
                    },
                   save_epoch)

    def load_model(self, epoch=0, map_location=None):
        save_epoch = "{}/epoch-{}.params".format(self.model_folder, str(epoch))
        print("loading %s" % save_epoch)
        checkpoint = torch.load(save_epoch, map_location=map_location)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("loaded: epoch-%d loss-%.4f" % (checkpoint['epoch'], checkpoint['loss']))

    def optimize(self, clip=5):
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.loss = 0.0

    # def evaluate_epoch(self):
    #     return
    #
    # def train_epoch(self):
    #     return
    #
    # def evaluate_batch(self):
    #     return
    #
    # def train_batch(self):
    #     return
