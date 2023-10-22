from torch import device

class CONSTANT():
    def __init__(self):
        self.epochs = 100
        self.lr = 1e-4
        self.wd = 1e-4
        self.bs = 16
        self.nw = 16
        self.pm = True
        self.milestones = [30, 80]
        self.gamma = 0.5
        self.patience = 5
        self.verbose = 1
        self.device = device('cuda:0')

        self.data_path_train = '../data/separated/train/PKS'
        self.data_path_valid = '../data/separated/valid/PKS'
        self.data_path_test = '../data/separated/test'
        self.data_path_artist = '../data/artist.txt'

        