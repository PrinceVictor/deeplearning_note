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

        nn.init.xavier_uniform_(self.layer1.conv1.weight)
        nn.init.xavier_uniform_(self.layer2.conv2.weight)

        self.outpool_size = [4, 2, 1]

        self.layer4 = nn.Sequential(
            nn.Linear((32*4*4 + 32*2*2 + 32*1*1), 256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            # nn.Linear(256, 10)
            nn.Linear(256, 10)
        )

    def sppmodule(self, input, previous_conv_size):
        pre_h, pre_w = previous_conv_size
        for i in range(len(self.outpool_size)):
            h_wid = int(np.ceil(pre_h / self.outpool_size[i]))
            w_wid = int(np.ceil(pre_w / self.outpool_size[i]))
            h_str = int(np.floor(pre_h / self.outpool_size[i]))
            w_str = int(np.floor(pre_w / self.outpool_size[i]))

            max_pool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_str, w_str))
            temp = max_pool(input)
            # print('temp size {}'.format(temp.shape))

            if(i == 0):
                spp = temp.view(input.size(0), -1)
            else:
                spp = torch.cat((spp, temp.view(input.size(0), -1)), 1)
        return spp

    def forward(self, input):

        layer1 = self.layer1(input)
        # print('layer1 size {}', layer1.shape)
        layer2 = self.layer2(layer1)
        # print('layer2 size {}', layer2.shape)
        spp = self.sppmodule(layer2, [layer2.size(2), layer2.size(3)])
        # print('spp size {}' .format(spp.shape))

        output = self.layer4(spp)

        return output



