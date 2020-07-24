# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/8/25 9:10
@Describe：

"""

import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import Linear

# The x object is a list.
x = [12, 23, 34, 45, 56, 67, 78]
print(torch.is_tensor(x))  # False
print(torch.is_storage(x))  # False

# s create an object that contains random numbers from Torch
y = torch.randn(1, 2, 3, 4, 5)
print(torch.is_tensor(y))  # True
print(torch.is_storage(y))  # False
print(torch.numel(y))  # 返回元素数目  120 the total number of elements in the input tensor
print(y)

print('*************' * 5)
print(torch.zeros(4, 4))
x1 = np.array(x)
print(x1)
print(torch.from_numpy(x1))
print(torch.linspace(2, 10, steps=25))


def forward(x):
    print('ffffffffff')
    return x * w


w = Variable(torch.Tensor([1.0]), requires_grad=True)
