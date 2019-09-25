import d2lzh as dl
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import mxnet.autograd as ag
import mxnet as mx
import os
import sys
from mxnet.gluon import data as gdata

f_mnist_train = gdata.vision.FashionMNIST(train=True)
f_mnist_test = gdata.vision.FashionMNIST(train=False)
transformer = gdata.vision.transforms.ToTensor()


# 混合式编程！！！命令式编程运行速度慢
# 符号式编程：
# 1. 定义计算流程；
# 2. 把计算流程编译成可执行的程序；
# 3. 给定输入，调用编译好的程序执行
# 高效 易移植
# 当我们调用hybridize函数后，Gluon会转换成依据符号式编程的方式执行

# 采用混合式编程，用HbridBlock类定义卷积网络
# 只有继承HybridBlock类的层才会被优化计算。例如，HybridSequential类和Gluon提供的Dense类
# 都是HybridBlock类的子类，它们都会被优化计算。如果一个层只是继承自Block类而不是HybridBlock类，那么它将不会被优化。


class CNN(nn.HybridBlock):
    def __init__(self):
        super(CNN, self).__init__()
        with self.name_scope():
            ks = [3, 3, 3, 3, 3, 3, 3]
            ps = [1, 1, 1, 1, 1, 1, 0]
            ss = [1, 1, 1, 1, 1, 1, 1]
            nm = [32, 64, 64, 64, 64, 64, 64]
            blocks = []
            for nlayer, (k, p, s, n) in enumerate(zip(ks, ps, ss, nm)):
                conv = nn.Conv2D(channels=n, padding=p, strides=s, kernel_size=k)
                activ = nn.LeakyReLU(alpha=.1)
                bn = nn.BatchNorm()
                blocks.append(conv)
                blocks.append(bn)
                blocks.append(activ)
                if nlayer in (0, 1):
                    blocks.append(nn.MaxPool2D(pool_size=(2, 2), prefix="pooling{}".format(nlayer)))

            self.cnn = nn.HybridSequential()
            self.cnn.add(*blocks)
            self.dense0 = nn.Dense(10)

    # 正向传播
    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        x = x.reshape(0, 0, -1)
        x = x.mean(axis=2)
        return self.dense0(x)


transformers = gdata.vision.transforms.ToTensor()


def accuracy(y_hat, y): return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


# 定义数据加载函数
def getDataLoader():
    return mx.gluon.data.DataLoader(f_mnist_train.transform_first(transformers), batch_size=128, shuffle=True), \
           mx.gluon.data.DataLoader(f_mnist_test.transform_first(transformers), batch_size=128, shuffle=False)


net = CNN()
net.initialize(init.Normal(sigma=0.01))
batch_size = 128
train_iter, test_iter = getDataLoader()
loss = gloss.SoftmaxCrossEntropyLoss()
net.collect_params().reset_ctx(mx.cpu(0))
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epochs = 30

# training
for epoch in range(1, num_epochs + 1):
    train_l_sum = 0
    train_acc_sum = 0
    for X, y in train_iter:
        X = X.as_in_context(mx.cpu(0))
        y = y.as_in_context(mx.cpu(0))
        with ag.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()

        trainer.step(batch_size)
        train_l_sum += l.mean().asscalar()
        train_acc_sum = train_acc_sum + accuracy(y_hat, y)
    net.collect_params().reset_ctx(mx.cpu(0))
    test_acc = dl.evaluate_accuracy(test_iter, net)
    net.collect_params().reset_ctx(mx.cpu(0))
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (epoch, train_l_sum / len(train_iter),
             train_acc_sum / len(train_iter), test_acc))
