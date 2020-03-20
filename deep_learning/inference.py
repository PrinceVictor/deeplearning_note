import torch
import torch.nn as nn
import torch.utils.data
import module.LeNet as LeNet
from PIL import Image
import matplotlib.pyplot as plt
from dataload import preprocess, DataLoader
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description='deep learning network practice')
parser.add_argument('--model-name', default='relu_lenet50.pkl',
                    help='model name')
parser.add_argument('--load-modelpath', default='/home/victor/darling/deeplearning_note/model_para',
                    help='load model path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--input', required=True,
                    help='inference input')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if __name__ == '__main__':


    if args.cuda == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = LeNet.LeNet()
    model.to(device)

    load_path = os.path.join(args.load_modelpath, args.model_name)
    print('load model path: {}\n'
           'device {}' .format(load_path, device))

    state_dict = torch.load(load_path)

    for k, v in state_dict.items():
        name = k
        print('name : {} size {}' .format(name, v.shape))
    # model.load_state_dict(state_dict)

    # input = args.input
    # img = Image.open(input).convert('L').resize((32, 32))
    # # img.show()
    # img = np.array(img)
    # # plt.imshow(img, cmap='gray')
    # # plt.show()
    # img = 255 - torch.Tensor(img).float().to(device)
    # img = torch.unsqueeze(img.unsqueeze(0), 0)
    # print('img size {} type {}'.format(img.shape, img.type()))
    #
    # with torch.no_grad():
    #     model.eval()
    #     output = (model(img)).cpu()
    # print('output size {}'.format(output.shape))
    # _, output =torch.max(torch.squeeze(output), -1)
    # print('output {}' .format(output))






