import numpy as np
from mxnet import nd

x = nd.arange(12)

x = x.reshape(1,3,2,2)
# x = x.reshape(3,4)

y = x.reshape(0, 0, -1)


z = y.mean(axis=2)
print(x)
# print(z)
print(y)

print("a" , y.shape)
