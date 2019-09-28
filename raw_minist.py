import d2lzh as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import mxnet.autograd as ag
import mxnet as mx
import os
mnist = mx.test_utils.get_mnist()
class CRNN(nn.HybridBlock):
    def __init__(self):
        super(CRNN, self).__init__()
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

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        x = x.reshape(0, 0, -1)
        x = x.mean(axis=2)
        return self.dense0(x)

def getDataLoader():
    class MinistTrainDataset():
        def __init__(self):
            self.train_data = mnist['train_data']
            self.train_label = mnist['train_label']
            self.test_data = mnist['test_data']
            self.test_label = mnist['test_label']

        def __len__(self):
            return len(self.train_data)

        def __getitem__(self, item):
            return self.train_data[item], self.train_label[item]

    class MinistTestDataset():
        def __init__(self):
            self.train_data = mnist['train_data']
            self.train_label = mnist['train_label']
            self.test_data = mnist['test_data']
            self.test_label = mnist['test_label']

        def __len__(self):
            return len(self.test_data)

        def __getitem__(self, item):
            return self.test_data[item], self.test_label[item]

    class DirDataSet(object):
        def __init__(self, root):
            self.objs = []
            for r, _, names in os.walk(root):
                for name in names:
                    self.objs.append(os.path.join(r, name))

        def __len__(self):
            return len(self.objs)

    class Char70KDataset(DirDataSet):
        def __getitem__(self, item):
            path = self.objs[item]
            label = path.split("/")[-2]
            label = int(label)
            image = mx.image.imread(path, flag=0).astype('f')
            image = mx.image.imresize(image, 28, 28)
            image = image.transpose(axes = (2, 0, 1)) / 255.0
            image = 1 - image
            return image.asnumpy(), label

    class DigitalDataset(DirDataSet):
        def __getitem__(self, item):
            path = self.objs[item]
            label = path.split("/")[-2]
            label = int(label)
            image = mx.image.imread(path, flag=0).astype('f')
            image = mx.image.imresize(image, 28, 28)
            image = image.transpose(axes = (2, 0, 1)) / 255.0
            return image.asnumpy(), label


    class ConcatDataset(object):
        def __init__(self, *datasets):
            self.datasets = datasets

        def __len__(self):
            return sum(len(x) for x in self.datasets)

        def __getitem__(self, idx):
            start_idx = 0
            for da in self.datasets:
                if start_idx <= idx < start_idx + len(da):
                    return da[idx - start_idx]
                start_idx += len(da)

    minist_dataset = MinistTrainDataset()
    train_dataset = minist_dataset
    return mx.gluon.data.DataLoader(train_dataset, batch_size=128, shuffle=True),\
            mx.gluon.data.DataLoader(MinistTestDataset(), batch_size=128, shuffle=False),

def accuracy(y_hat, y): return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

net = CRNN()
net.initialize(init.Normal(sigma=0.01))

batch_size = 128
# train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
train_iter, test_iter = getDataLoader()
loss = gloss.SoftmaxCrossEntropyLoss()
net.collect_params().reset_ctx(mx.gpu(0))
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epochs = 1
"""Train and evaluate a model on CPU."""
for epoch in range(1, num_epochs + 1):
    train_l_sum = 0
    train_acc_sum = 0
    for X, y in train_iter:
        X = X.as_in_context(mx.gpu(0))
        y = y.as_in_context(mx.gpu(0))
        with ag.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()

        trainer.step(batch_size)
        train_l_sum += l.mean().asscalar()
        train_acc_sum += accuracy(y_hat, y)

    net.collect_params().reset_ctx(mx.cpu())
    test_acc = gb.evaluate_accuracy(test_iter, net)
    net.collect_params().reset_ctx(mx.gpu(0))
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
          % (epoch, train_l_sum / len(train_iter),
             train_acc_sum / len(train_iter), test_acc))
    net.collect_params().save("./output_params/%d.params" % (epoch))
