from utils.log import Log
from models.ToyModel import ToyModel
from models.ModelRunner import ModelRunner

if __name__ == '__main__':
    # logger = Log('test')
    # for i in range(10):
    #     logger.write(str(i))
    #     print(i)
    #     exit()
    # logger.f.close()

    runner = ModelRunner('test', model=ToyModel())
    runner.set_optimizer()
    runner.save_model(0, 0)

    runner2 = ModelRunner('test', model=ToyModel())
    runner2.set_optimizer()
    runner2.load_model(0)
    print(runner2.model)
    print(runner2.optimizer)
