import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import matplotlib.pyplot as plt
import numpy as np

from dataload import preprocess, DataLoader
import module.LeNet as LeNet
import module.ownnet as ownet

import os
import argparse

parser = argparse.ArgumentParser(description='deep learning network practice')
parser.add_argument('--model-name', required=True,
                    help='model name')
parser.add_argument('--pretrained', default=None,
                    help='pre trained model')
parser.add_argument('--last-epoch', default=-1, type=int,
                    help='last-epoch')
parser.add_argument('--figure-name', default=None,
                    help='figure name')
parser.add_argument('--model-savepath', default='/home/victor/darling/deeplearning_note/model_para',
                    help='model save path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--epoch', default=10, type=int,
                    help='enables CUDA training')
parser.add_argument('--batchsize', default=128, type=int,
                    help='batch size')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

def trainnet(model, device, train_loader, optimizer, lr_shedule, epoch, loss_list):
    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        predict = model(data)

        class_softmax = F.softmax(predict, dim=1).gather(1, target.view(-1, 1))
        class_logsoft = class_softmax.log()
        gamma = 2
        class_loss = -torch.mul(torch.pow(1-class_softmax, gamma), class_logsoft)
        class_loss = class_loss.mean()

        # class_loss = F.cross_entropy(predict, target)

        class_loss.backward()
        optimizer.step()
        loss_list.append(class_loss.cpu().data.item())

        pred = predict.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)  '
                  'Loss: {:.6f}  '
                  'Learn rate: {:.6f}]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), class_loss.item(),
                optimizer.param_groups[0]['lr']))

    lr_shedule.step()
    accuracy = 100. * correct / len(train_loader.dataset)
    print('Train set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(train_loader.dataset), accuracy))

def testnet(model, device, test_loader, accuracy_list):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_list.append(accuracy)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))


if __name__ == '__main__':

    EPOCHS = args.epoch
    Batchsize = args.batchsize
    model_name = args.model_name + str(EPOCHS)

    if args.cuda == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    save_path = os.path.join(args.model_savepath, model_name + '.pkl')
    print('model save path: {}\n'
           'train epoch {}\n'
           'device {}' .format(save_path, EPOCHS, device))

    train_process = preprocess.get_transform((28, 28), argument=False)
    test_process = preprocess.get_transform((28, 28), argument=False)

    trainset = DataLoader.fashionmnist_dataset(train_process, train=True)
    testset = DataLoader.fashionmnist_dataset(test_process, train=False)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=Batchsize,
                                              shuffle=True,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                              batch_size=Batchsize,
                                              shuffle=False,
                                              num_workers=2)

    # model = LeNet.LeNet()
    model = ownet.Ownnet()
    model.to(device)

    for k, v in model.state_dict().items():
        name = k
        print('name : {} size {}' .format(name, v.shape))

    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)
        model.load_state_dict(state_dict)

    if args.last_epoch != -1:
        start_epoch = args.last_epoch + 1
    else:
        start_epoch = 1

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters {}' .format(total_params))

    optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': 0.001}], lr=0.001, momentum=0.8)
    # optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.001}],
    #                        lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_shedule = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1, last_epoch=args.last_epoch)
    lr_shedule.step()

    loss_list = []
    acc_list = []

    for epoch in range(start_epoch, EPOCHS + 1):
        trainnet(model, device, trainloader, optimizer, lr_shedule, epoch, loss_list)
        testnet(model, device, testloader, acc_list)

    figure_path = args.figure_name
    x_loss_list = np.linspace(0, EPOCHS, len(loss_list))
    x_acc_list = np.linspace(0, EPOCHS, len(acc_list))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x_loss_list, loss_list, linewidth=2, color='r', label='loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.plot(x_acc_list, acc_list, linewidth=2, color='g', label='accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig("result/" + model_name)
    plt.show()

    torch.save(model.state_dict(), save_path)








