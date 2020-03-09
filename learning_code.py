import numpy as np

x = np.arange(6)
print('x: {}'.format(x))
print('x shape: {}'.format(x.shape))

x = x.reshape(3, 2)
print('x: {}'.format(x))
print('x shape: {}'.format(x.shape))

# 创建数组
# numpy.zeros(shape, dtype = float, order = 'C')

a = np.zeros((2, 3, 2), dtype=np.int)
print('a: {}'.format(a))
print('a shape: {}'.format(a.shape))

# np.asarray

a = np.random.randint(0, 10, 10)
a = np.arange(10)
print('a: {}'.format(a))
# 从索引 2 开始到索引 7 停止，间隔为 2
print('a[2:7:2]: {}'.format(a[2:7:2]))
print('a[5:]: {}'.format(a[5:]))
print('a[5:9]: {}'.format(a[5:9]))
#冒号 : 的解释：如果只放置一个参数，如 [2]，将返回与该索引相对应的单个元素。
#如果为 [2:]，表示从该索引开始以后的所有项都将被提取。如果使用了两个参数，
# 如 [2:7]，那么则提取两个索引(不包括停止索引)之间的项。

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
# 从某个索引处开始切割
print('从数组索引 a[1:] 处开始切割')
print(a[1:])

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print (a[..., 1])   # 第2列元素
print (a[1, ...])   # 第2行元素
print (a[..., 1:])  # 第2列及剩下的所有元素

import numpy as np

x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]]
print(y)
# [1  4  5]

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print('我们的数组是：')
print(x)
print('\n')
rows = np.array([[0, 0], [3, 3]])
cols = np.array([[0, 2], [0, 2]])
y = x[rows, cols]
print('这个数组的四个角元素是：')
print(y)

# 我们的数组是：
# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
# 这个数组的四个角元素是：
# [[ 0  2]
#  [ 9 11]]


a = np.array([[1,2,3], [4,5,6],[7,8,9]])
b = a[1:3, 1:3]
c = a[1:3,[1,2]]
d = a[...,1:]
print(b)
print(c)
print(d)

x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
print ('我们的数组是：')
print (x)
print ('\n')
# 现在我们会打印出大于 5 的元素
print ((x > 5) + 1)
print  ('大于 5 的元素是：')
print (x[x >  5])
print  ('大于 5 的元素赋值0：')
x[x >  5] = 0
print (x)

x=np.arange(6).reshape(2,3)
print(x)

for row in x:
    print(row)

print(x.T) # 转置
for row in x.T:
    print(row)

a=np.random.rand(5)
print(a)
# [ 0.64061262  0.8451399   0.965673    0.89256687  0.48518743]

print(a[-1]) ###取最后一个元素
# [0.48518743]

print(a[:-1])  ### 除了最后一个取全部
# [ 0.64061262  0.8451399   0.965673    0.89256687]

print(a[::-1]) ### 取从后向前（相反）的元素
# [ 0.48518743  0.89256687  0.965673    0.8451399   0.64061262]

print(a[2::-1]) ### 取从下标为2的元素翻转读取
# [ 0.965673  0.8451399   0.64061262]
# x=np.arange(32).reshape((8,4))
# print(x)
# print (x[[4,2,1,7]])

import numpy as np
# 坐标向量
a = np.arange(10)
# 坐标向量
b = np.arange(5)
# 从坐标向量中返回坐标矩阵
# 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
col, row = np.meshgrid(a,b)
print(row, '\n', col)
t = np.stack([col, row])
print("------------")
print(t)
print(t.shape)
t = t.reshape(2, -1)
print(t)
print(t.shape)
print(t.T)

