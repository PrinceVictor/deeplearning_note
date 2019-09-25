#matplotlib inline
import d2lzh as d2l
from mxnet.gluon import data as gdata
import matplotlib.pyplot as plt
import sys
import time
import os

print(os.path.abspath('./'))

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

#
# print(len(mnist_train))
# print(len(mnist_test))
#
# X, y = mnist_train[10:20]
# d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))
# plt.savefig('./image_show.png')
# plt.show()

# flg1 = plt.figure(1)
# plt.imshow(mnist_train[0][0].reshape(28,28).asnumpy())
# plt.savefig('./image_show1')
# flg2 = plt.figure(2)
# plt.imshow(mnist_train[1][0].reshape(28,28).asnumpy())
# plt.savefig('./image_show2')
# plt.show()


print("success!")
