import torch
import  numpy as np

a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
print('a is {}' .format(a))
print('a size {}' .format(a.size()))
print('a size {}' .format(a.size(0)))

b = a.numpy()
# c = torch.from_numpy(b)
c = torch.Tensor(b)
print('b is {}' .format(b))
print('c is {}' .format(c))

# if torch.cuda.is_available():
#     c = c.cuda()
#     print('c cuda {}' .format(c))
row = np.arange(2)
col = np.arange(3)
disp = np.random.randint(1,10,(2,3))
col, row = np.meshgrid(col, row)
matrix = np.stack((col, row, disp))
print('matrix \n{}' .format(matrix))
print(matrix.shape)

# matrix = matrix.reshape(3, -1)
# print('matrix reshape\n{}' .format(matrix))

x = torch.Tensor(matrix)
print('x is \n{}' .format(x))
print('x size {}' .format(x.size()))
#
# print(x[:, -1])
### torch.view similar with numpy.reshape()
print((x.view(3, -1)))

# variable wae deprecated
# x = torch.ones(2, 2, requires_grad= True)
# print('x is \n{}' .format(x))
#
# y = x + 2
# print('y is \n{}' .format(y))

x = torch.randn(3, requires_grad=True)

y = x * 2
y = y.sum()

print(x)
print(y)

y.backward()
print(x.grad)

