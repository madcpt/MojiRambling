from model.BertEnc import BertEnc
from utils.BertRunner import BertRunner
from preprocess.make_vocab import Lang
from tqdm import tqdm
import torch


if __name__ == '__main__':
    device = torch.device('cuda:0')
    # lang = Lang('SS-Youtube', device)
    lang = Lang('PsychExp', device)
    # lang.read_raw()
    train, dev, test, word2index = lang.load_preprocessed(batch_size=8)

    # runner = BertRunner('baseline', model=BertEnc(len(word2index), 'SS-Youtube', True, False), device=device)
    runner = BertRunner('baseline', model=BertEnc(len(word2index), 'PsychExp', True, False, 7), device=device)

    runner.set_optimizer(1e-3)
    # runner.save_model(0, 0)
    runner.evaluate_epoch(dev, 0, 'dev')
    runner.evaluate_epoch(test, 0, 'test')
    for epoch in range(100):
        train, dev, test, _ = lang.load_preprocessed(4)
        runner.train_epoch(train, epoch)
        runner.evaluate_epoch(dev, epoch, 'dev')
        runner.evaluate_epoch(test, epoch, 'test')
