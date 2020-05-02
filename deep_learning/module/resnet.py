import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict

class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()

        self.layer1 = nn.Sequential(OrderedDict([
            # (28-3 + 4)/1 + 1= 27 16*30*30
            ('conv1', nn.Conv2d(1, 16, kernel_size=3, padding=2)),
            ('batchnorm1', nn.BatchNorm2d(16)),
            ('relu1', nn.ELU()),
            # nn.ReLU(),
            # 16*15*15
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        self.layer2 = nn.Sequential(OrderedDict([
            # (15-3+4)/2 + 1= 12 32*9*9
            ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2)),
            ('batchnorm2', nn.BatchNorm2d(32)),
            # nn.ReLU(),
            ('relu2', nn.ELU()),
            # 32*3*3
            # ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

        self.layer4 = [nn.Sequential(
            nn.Linear((32*4*4 + 32*2*2 + 32*1*1), 256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            # nn.Linear(256, 10)
            nn.Linear(256, 10)
        )]

        self.layer5= nn.ModuleList(self.layer4)

    def forward(self, input):

        layer1 = self.layer1(input)
        # print('layer1 size {}', layer1.shape)
        layer2 = self.layer2(layer1)
        # print('layer2 size {}', layer2.shape)
        # print('spp size {}' .format(spp.shape))

        output = self.layer4(layer2)

        return output

if __name__ == '__main__':

        resnet = Resnet()
        print(resnet)
        for k, v in resnet.state_dict().items():
            name = k
            print('name: {} size: {}'.format(name, v.shape))
