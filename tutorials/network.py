import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd.function import Function

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        # print(x.size())
        num_features = 1
        for s in size:
            num_features *= s
        # print(num_features)
        return num_features

if __name__ == '__main__':
    net = Network()
    print(net)
    # params = list(net.parameters())
    # print(len(params)
    # print(params[0].size())
    optimzer = optim.SGD(net.parameters(), lr=0.01)
    optimzer.zero_grad()

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print('out size {}' .format(out.size()))
    print(out)

    # net.zero_grad()
    # out.backward(torch.randn(1,10))
    target = torch.randn(1, 10)
    print('target size{}'.format(target.size()))
    target = target.view(1, -1)
    print('target {}' .format(target))

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print('loss: {}'.format(loss.grad_fn))

    net.zero_grad()
    loss.backward()

    optimzer.step()





