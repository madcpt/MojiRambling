# from model.BaseLine import BaseLine as moji
from model.deepmoji import DeepMoji as moji
from utils.BaseLineRunner import BaseLineRunner
# from preprocess.make_vocab import Lang as dataset
from preprocess.pre_processing import origin_dataset as dataset
from tqdm import tqdm
import torch


if __name__ == '__main__':
    device = torch.device('cuda:0')

    lang = dataset('data1_170000', device)
    # lang.read_raw()
    train, dev, test, word2index = lang.load_preprocessed(4)

    # runner = BaseLineRunner('deepmoji', model=moji(len(word2index), load_emb=True, emb_fixed=False, dim=400), device=device)
    runner = BaseLineRunner('deepmoji', model=moji(len(word2index), load_emb=True, emb_fixed=False, dim=400), device=device)

    # # lang = Lang('SS-Youtube', device)
    # lang = Lang('PsychExp', device)
    # # lang.read_raw()
    # train, dev, test, word2index = lang.load_preprocessed(4)
    #
    # # runner = BaseLineRunner('baseline', model=BaseLine(len(word2index), 'SS-Youtube', True, False), device=device)
    # runner = BaseLineRunner('baseline', model=BaseLine(len(word2index), 'PsychExp', True, False, 7), device=device)

    runner.set_optimizer()
    # runner.save_model(0, 0)
    runner.evaluate_epoch(dev, 0, 'dev')
    runner.evaluate_epoch(test, 0, 'test')
    for epoch in range(20):
        train, dev, test, _ = lang.load_preprocessed(128)
        runner.train_epoch(train, epoch)
        runner.evaluate_epoch(dev, epoch, 'dev')
        runner.evaluate_epoch(test, epoch, 'test')
