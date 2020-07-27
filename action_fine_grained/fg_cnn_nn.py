# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/27 15:38
@Describe：

"""


import torch
from torch import nn

import torch.nn.functional as F


class FG_Net_CNN(nn.Module):
    def __init__(self):
        super(FG_Net_CNN, self).__init__()
        self.row = 40
        self.col = 6
        self.cov_depth = [4]  # 2个卷积层的厚度

        # input data size = 1* row*col
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=1, padding=2),  # 4*40*42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # 4*(40/2)*(6/2) = 4*20*3
        )

        # 第一个线性层
        self.layer1 = nn.Linear(240, 100)
        # Relu层
        self.layer2 = nn.ReLU()
        # 分类线性层
        self.layer3 = nn.Linear(100, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        out = F.log_softmax(x, dim=1)
        return out

if __name__ == '__main__':
    cnn = FG_Net_CNN()
    print(cnn)


