import torch
import argparse


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.save_model = True

        self.parser = argparse.ArgumentParser(description='Fine-Tune')

    def parse_args(self):
        self.parser.add_argument('-mode', '--mode', help='pretrain/transfer', required=True, default="transfer")
        self.parser.add_argument('-train', '--train', help='0: test, 1: train', required=True, default=0)
        self.parser.add_argument('-model', '--model', help='deepmoji/transformermoji/bert/bertmoji', required=True, default="deepmoji")
        self.parser.add_argument('-ds', '--dataset', help='dataset', required=True, default="SS-Youtube")
        self.parser.add_argument('-n', '--nb_classes', help='nb_classes', required=True, default=2)
        self.parser.add_argument('-cuda', '--use_cuda', help='0: use cpu, 1: use cuda', required=True, default=1)
        self.args = vars(self.parser.parse_args())

        if int(self.args['use_cuda']) == 0:
            self.device = torch.device('cpu')
        self.mode = self.args['mode']
        self.model = self.args['model']
        self.dataset = self.args['dataset']
        self.nb_classes = self.args['nb_classes']
        self.train = int(self.args['train']) == 1


config = Config()
config.parse_args()

if __name__ == '__main__':
    print(config.mode)

