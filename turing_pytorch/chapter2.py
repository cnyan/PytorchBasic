# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
计算梯度下降
预测房价
"""

from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt

x = Variable(torch.linspace(0, 100), requires_grad=True)
# print(x.detach().numpy())
rand = Variable(torch.randn(100)) * 10
y = x + rand
# print(y.detach().numpy())
print(torch.mean(rand))
print(torch.std(rand))

x_train = x[:-10]
x_test = x[-10:]
y_train = y[:-10]
y_test = y[-10:]

a = Variable(torch.rand(1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)
learning_rate = 1e-4

# 对a,b迭代计算
for i in range(1000):
    # Pytorch 中 expand, expand_as是共享内存的，只是原始数据的一个视图,把一个tensor变成(扩维)和函数括号内一样形状的tensor
    predictions = a.expand_as(x_train) * x_train + b.expand_as(x_train)
    # predictions = a * x_train + b
    loss = torch.mean((predictions - y_train) ** 2)
    # print('loss:' + str(loss))

    loss.backward()
    # .使用add_()对原始的tensor的每一个数值进行加的操作
    a.data.add_(-learning_rate * a.grad.data)
    b.data.add_(-learning_rate * b.grad.data)

    a.grad.data.zero_()
    b.grad.data.zero_()
print('loss:' + str(loss))
print(a.data)
print(b.data)

plt.figure(figsize=(10, 8))
plt.plot(x_train.detach().numpy(), y_train.data.numpy(), 'o')
plt.plot(x_train.data.numpy(), (a.data.numpy() * x_train.data.numpy() + b.data.numpy()))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 预测
predictions = a*x_test+b
print(predictions)

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/9/2 14:39
@Describe：



import sys

print('HEO=LLO')
output = sys.stdout
outputfile = open("D:\\print.txt", "a")
sys.stdout = outputfile
type = sys.getfilesystemencoding()#python编码转换到系统编码输出

print('测试 '*3)
print('测试 '*3)
print('测试 '*3)
"""

"""
import sys
import os


class Logger(object):
    def __init__(self, filename="D://Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('D://a.txt')

print(path)
print(os.path.dirname(__file__))
print('------------------')
"""
