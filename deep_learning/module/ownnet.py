import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import OrderedDict

class Ownnet(nn.Module):

    def __init__(self):
        super(Ownnet, self).__init__()

        self.layer1 = nn.Sequential(OrderedDict([
            # (28-3 + 2)/1 + 1= 27 16*28*28
            ('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1)),
            ('relu1', nn.ELU()),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # 16*14*14
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            # (14-3)/1 + 1= 12 16*12*12
            ('conv2', nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            ('relu2', nn.ELU()),
            # 16*6*6
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))

        nn.init.xavier_uniform_(self.layer1.conv1.weight)
        nn.init.xavier_uniform_(self.layer1.conv2.weight)

        # self.layer2 = nn.Sequential(
        #     # (9-3)/2 + 1= 4 32*4*4
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2),
        #     # nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     # 32*3*3
        #     # nn.MaxPool2d(kernel_size=2, stride=1)
        # )
        #
        # self.layer3 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=4, stride=2)
        #     # (3-3)/1 + 1= 12 32*1*1
        #     # nn.Conv2d(32, 32, kernel_size=3, stride=1),
        #     # nn.BatchNorm2d(32),
        #     # nn.ReLU()
        # )

        # window1 = np.ceil(12/4)
        # stride1 = np.floor(12/4)
        #
        # self.layer2 = nn.Sequential(
        #     nn.MaxPool2d(window1, stride=stride1),
        # )

        # window2 = np.ceil(4/1)
        # stride2 = np.floor(4/1)
        # self.layer3 = nn.Sequential(
        #     nn.MaxPool2d(window2, stride=stride2),
        # )

        self.layer4 = nn.Sequential(
            # nn.Linear((9*9*16 + 4*4*32 + 1*1*32), 256),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(256, 10)
            nn.Linear(32*6*6, 10)
        )


    def forward(self, input):

        layer1 = self.layer1(input)
        # print('layer1 size {}', layer1.shape)
        # layer2 = self.layer2(layer1)
        # print('layer2 size {}', layer2.shape)
        # layer3 = self.layer3(layer2)
        # print('layer3 size {}', layer3.shape)

        batch_size = layer1.size(0)
        # concat = torch.cat((layer1.view(batch_size, -1), layer2.view(batch_size, -1)), 1)
        # concat = torch.cat((concat, layer3.view(batch_size, -1)), 1)


        output = self.layer4(layer1.view(batch_size, -1))
        return output