import torch

x = torch.rand(5, 3)

print('x: {}'.format(x))

print('x size: {}'.format(x.size()))

print('x check: {}'.format(x.shape[0]))
