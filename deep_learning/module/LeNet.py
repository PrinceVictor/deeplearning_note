import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.activate = nn.ReLU()
        # self.activate = nn.LeakyReLU()

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):

        temp = self.pool(self.activate(self.conv1(input)))
        temp = self.pool(self.activate(self.conv2(temp)))
        temp = temp.view(-1, 16*5*5)
        temp = self.activate(self.fc1(temp))
        temp = self.activate(self.fc2(temp))
        output = self.fc3(temp)
        return output