import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import mxnet as mx
import mxnet.autograd as ag
import time

class CRNN(nn.HybridBlock):
    def __init__(self):
        super(CRNN, self).__init__()
        with self.name_scope():
            kernals = [11, 5, 3, 3, 3]
            paddings = [0, 2, 1, 1, 1]
            strides = [4, 1, 1, 1, 1]
            channels = [96, 256, 384, 384, 256]
            blocks = []
            for nlayer, (k, p, s, n) \
                    in enumerate(zip(kernals, paddings, strides, channels)):
                conv = nn.Conv2D(channels=n,
                                 kernel_size=k,
                                 padding=p,
                                 strides=s,
                                 activation='relu')
                blocks.append(conv)
                if nlayer in (0, 1, 4):
                    blocks.append(nn.MaxPool2D(pool_size=3, strides=2))

            self.cnn = nn.HybridSequential()
            self.cnn.add(*blocks)

            self.cnn.add(nn.Dense(4096,
                                  activation='relu'),
                         nn.Dropout(0.5))
            self.cnn.add(nn.Dense(4096,
                                  activation='relu'),
                         nn.Dropout(0.5))
            self.cnn.add(nn.Dense(10))

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        return x

def accuracy(y_hat, y): return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

batch_size = 128

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

loss = gloss.SoftmaxCrossEntropyLoss()

net = CRNN()
net.initialize(init.Normal(sigma=0.01))
net.collect_params().reset_ctx(mx.gpu(0))

trainer = gluon.Trainer(net.collect_params(),
                        'sgd',
                        {'learning_rate': 0.1})

num_epochs = 5

for epoch in range(1, num_epochs + 1):
    train_loss_sum = 0
    train_acc_sum = 0
    start = time.time()
    for X, y in train_iter:
        X = X.as_in_context(mx.gpu(0))
        y = y.as_in_context(mx.gpu(0))
        with ag.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()

        trainer.step(batch_size)
        train_loss_sum += l.mean().asscalar()
        train_acc_sum += accuracy(y_hat, y)

    test_acc = d2l.evaluate_accuracy(test_iter, net, ctx=mx.gpu(0))

    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time cost %.3f'
          % (epoch, train_loss_sum / len(train_iter),
             train_acc_sum / len(train_iter), test_acc, time.time()-start))

net.collect_params().save("./output_params/fashion.params")
