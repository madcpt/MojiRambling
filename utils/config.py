import torch
import argparse


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda:0')
        # self.device = torch.device('cpu')

        self.save_model = True

        self.parser = argparse.ArgumentParser(description='Fine-Tune')

    def parse_args(self):
        self.parser.add_argument('-ds', '--dataset', help='dataset', required=True, default="SS-Youtube")
        self.parser.add_argument('-m', '--method', help='method', required=True, default="last")
        self.parser.add_argument('-n', '--nb_classes', help='nb_classes', required=True, default=2)


config = Config()

