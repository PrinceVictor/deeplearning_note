import torch
import torchvision.datasets

import numpy as np
from PIL import Image
import dataload.preprocess as preprocess

minist_classes = ('zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine')

# class Default_dataset(Dataset):
#
#     def __init__(self):
#
#     def __len__(self):
#
#     def __getitem__(self, item):

def mnist_dataset(transform, train=False):
    dataset = torchvision.datasets.MNIST(root='/home/victor/darling/deeplearning_note/dataset/mnist',
                                         train=train,
                                         download=False,
                                         transform=transform)

    return dataset


if __name__ == "__main__":
    image = (Image.open('../left.png'))
    print(image.size)
    process = preprocess.get_transform((32,32))
    image = process(image)
    print(image.size())