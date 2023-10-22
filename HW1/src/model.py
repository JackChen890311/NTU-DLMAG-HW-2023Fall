import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from constant import CONSTANT
from dataloader import MyDataloader


class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()

        # self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        # self.backbone = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        # self.backbone = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.DEFAULT)
        self.backbone = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        self.linear = nn.Linear(1000,20)

        # self.bn1 = nn.BatchNorm2d(1)
        # self.cnn0 = self.cnnBlock(0)
        # self.cnn1 = self.cnnBlock(1)
        # self.cnn2 = self.cnnBlock(2)
        # # self.cnn3 = self.cnnBlock(3)
        
        # self.rnn = nn.Sequential(
        #     nn.GRU(384, 32, 2, batch_first=True, dropout=0.3),
        # )

        # self.dense = nn.Sequential(
        #     nn.Linear(2432, 50),
        #     # nn.Softmax(dim=0)
        # )

    # def cnnBlock(self, i):
    #     num_filter_prev = [1, 64, 128, 128]
    #     num_filter_next = [64, 128, 128, 128]
    #     kernel_size = (3, 3)
    #     pool_size = [(2, 2), (4, 2), (4, 2), (4, 2), (4, 2)]

    #     layer = nn.Sequential(
    #             nn.Conv2d(num_filter_prev[i], num_filter_next[i], kernel_size),
    #             nn.ELU(),

    #             nn.BatchNorm2d(num_filter_next[i]),
    #             nn.MaxPool2d(pool_size[i]),
    #             nn.Dropout2d(0.1)
    #         )
    #     return layer
    

    def forward(self, x):
        # Return the output of model given the input x
        # x = self.bn1(x)
        # x = self.cnn0(x)
        # x = self.cnn1(x)
        # x = self.cnn2(x)
        # # x = self.cnn3(x)
        # x = torch.permute(x, (0, 3, 1, 2))
        # x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
        # x, h = self.rnn(x)
        # x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        # x = self.dense(x)
        
        # vocal = x[:,0]
        # music = x[:,1]
        # mix = (vocal + music * 0.5).unsqueeze(1)
        # x = torch.cat((x,mix), dim=1)
        x = self.backbone(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    C = CONSTANT()
    model = MyModel().to(C.device)
    print(model)
    
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    for x,y in dataloaders.loader['test']:
        x = x.to(C.device)
        yhat = model(x)
        print(x.shape,y.shape)
        print(yhat.shape)
        break