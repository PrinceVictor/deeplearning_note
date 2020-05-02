import torch
import numpy as np
import torch.nn as nn
import time

  # With square kernels and equal stride
input = torch.randn(2, 1, 6, 6)
print(input.size())
conv = nn.Conv2d(1, 1, (3, 3), stride=2, padding=0, bias=False)
output = conv(input)
print(output.size())
up_conv = nn.ConvTranspose2d(1, 1, (3,3), stride=2, padding=0, output_padding=1 ,bias=False)
input = up_conv(output)
print(input.size())
# print(input)
# input = input.repeat(2)
# print(input.size())
# print(input)
# x = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
# print(x.shape)

# print('i {}' .format(i) for i in range(4))
# a = np.random.randint(0,5, [1,1,5,2])
#
# a = a.astype(float)
# print(a.transpose().shape)
# # print(np.mean(a, axis=0))
# a = torch.tensor(a).float()
# # print(a.shape)
# # print(len(a))
# # print(a.mean(dim=1))
# # a = a.squeeze()
# a = torch.squeeze(a)
# print(a.shape)
# print(a)
# a = torch.max(a, 1)
# # a.max(1, keepdim=True)[1]
# # print(a.shape)
# print(a)
# print(a[1])

# a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
# print('a is {}' .format(a))
# print('a size {}' .format(a.size()))
# print('a size {}' .format(a.size(0)))
#
# b = a.numpy()
# # c = torch.from_numpy(b)
# c = torch.Tensor(b)
# print('b is {}' .format(b))
# print('c is {}' .format(c))
#
# # if torch.cuda.is_available():
# #     c = c.cuda()
# #     print('c cuda {}' .format(c))
# row = np.arange(2)
# col = np.arange(3)
# disp = np.random.randint(1,10,(2,3))
# col, row = np.meshgrid(col, row)
# matrix = np.stack((col, row, disp))
# print('matrix \n{}' .format(matrix))
# print(matrix.shape)
#
# # matrix = matrix.reshape(3, -1)
# # print('matrix reshape\n{}' .format(matrix))
#
# x = torch.Tensor(matrix)
# print('x is \n{}' .format(x))
# print('x size {}' .format(x.size()))
# #
# # print(x[:, -1])
# ### torch.view similar with numpy.reshape()
# print((x.view(3, -1)))
#
# # variable wae deprecated
# # x = torch.ones(2, 2, requires_grad= True)
# # print('x is \n{}' .format(x))
# #
# # y = x + 2
# # print('y is \n{}' .format(y))
#
# x = torch.randn(3, requires_grad=True)
#
# y = x * 2
# y = y.sum()
#
# print(x)
# print(y)
#
# y.backward()
# print(x.grad)

