import torch
from torch.utils.data import Dataset
import torchvision.datasets
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from preprocess import get_transform

# class Default_dataset(Dataset):
#
#     def __init__(self):
#
#     def __len__(self):
#
#     def __getitem__(self, item):







if __name__ == "__main__":
    image = (Image.open('left.png'))
    print(image.size)
    process = get_transform((32,32))
    image = process(image)
    print(image.size())