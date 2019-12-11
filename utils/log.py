import os


class Log(object):
    def __init__(self, model_name, model_folder='./save/test', overwrite=False):
        self.model_name = model_name
        self.model_folder = model_folder
        self.log_path = '%s/log.txt' % model_folder
        self.overwrite = overwrite
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        if not os.path.exists(self.log_path):
            self.overwrite = True
        if self.overwrite:
            self.f = open(self.log_path, 'w')
        else:
            self.f = open(self.log_path, 'a')
        self.f.write("*" * 50 + "\n")
        self.f.write("***%s***\n" % self.model_name)

    def __del__(self):
        self.f.close()
        print("log file exit: bye bye")

    def write(self, content):
        if type(content) is str:
            self.f.write("%s\n" % content)
        elif type(content) is list:
            for c in content:
                self.f.write("%s\t" % str(c))
            self.f.write("\n")
        else:
            self.f.write(str(content))
            self.f.write("\n")
        self.f.flush()
        return


if __name__ == '__main__':
    logger = Log('test')
    for i in range(10):
        logger.write(str(i))
        print(i)
