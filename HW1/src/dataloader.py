import os
import torch
import pickle as pk
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, mode):
        # Initialize your dataset object
        if mode == 'test':
            self.idx = sorted([int(a.split('.')[0]) for a in os.listdir(data_path) if a[-2:] == 'pk'])
            specList = []
            labelList = []
            for a in self.idx:
                with open(os.path.join(data_path, str(a)+'.pk'), 'rb') as f:
                    aspec = pk.load(f)
                    alabel = torch.ones((aspec.shape[0]))*a
                    specList.append(aspec)
                    labelList.append(alabel)
        else:
            self.artists = [a[:-3] for a in sorted(os.listdir(data_path))]
            specList = []
            labelList = []

            for id, a in enumerate(self.artists):
                with open(os.path.join(data_path, a+'.pk'), 'rb') as f:
                    aspec = pk.load(f)
                    alabel = torch.ones((aspec.shape[0]))*id
                    specList.append(aspec)
                    labelList.append(alabel)
            
        self.spec = torch.cat(specList)

        vocal = self.spec[:,0]
        music = self.spec[:,1]
        mix = (vocal + music * 0.5).unsqueeze(1)
        self.spec = torch.cat((self.spec,mix), dim=1)

        self.label = torch.cat(labelList)
        assert self.spec.shape[0] == self.label.shape[0]
        self.len = self.label.shape[0]

    def __len__(self):
        # Return the length of your dataset
        return self.len
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        return self.spec[idx], int(self.label[idx])


class MyDataloader():
    def __init__(self):
        super().__init__()
        self.C = CONSTANT()
        self.loader = {}
        with open(self.C.data_path_artist,'r') as f:
            self.artists = f.read().split('\n')

    def setup(self, types):
        print('Loading Data...')

        mapping = {
            'train':[self.C.data_path_train, True],
            'valid':[self.C.data_path_valid, False],
            'test' :[self.C.data_path_test, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, shuffle = mapping[name]
            self.loader[name] = self.loader_prepare(MyDataset(path, name), shuffle)
        
        if setupNames:
            print('Preparation Done! Use dataloader.loader[{type}] to access each loader.')
        else:
            print('Error: There is nothing to set up')

    def loader_prepare(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.C.bs,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm
        )


if __name__ == '__main__':
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])
    print(dataloaders.artists)

    for x,y in dataloaders.loader['test']:
        print(x.shape, y.shape)
        print(y)
        break