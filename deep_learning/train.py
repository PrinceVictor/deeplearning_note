import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from dataload import preprocess, DataLoader
import module.LeNet as LeNet
import torch.optim as optim

import os
import argparse

parser = argparse.ArgumentParser(description='deep learning network practice')
parser.add_argument('--model-name', default='LeNet.pkl',
                    help='model name')
parser.add_argument('--model-savepath', default='/home/victor/darling/deeplearning_note/model_para',
                    help='model save path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--epoch', default=10, type=int,
                    help='enables CUDA training')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

def trainnet(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def testnet(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    process = preprocess.get_transform((32, 32))

    trainset = DataLoader.mnist_dataset(process, train=True)
    testset = DataLoader.mnist_dataset(process, train=False)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=2)

    if args.cuda == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = LeNet.LeNet()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    EPOCHS = args.epoch

    save_path = os.path.join(args.model_savepath, args.model_name)
    print('model save path: {}\n'
           'train epoch {}\n'
           'device {}' .format(save_path, EPOCHS, device))

    for epoch in range(1, EPOCHS + 1):
        trainnet(model, device, trainloader, optimizer, epoch)
        testnet(model, device, testloader)

    torch.save(model.state_dict(), save_path)








