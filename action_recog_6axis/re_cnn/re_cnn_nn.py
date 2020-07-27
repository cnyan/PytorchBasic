# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/24 11:07
@Describe：
        自定义CNN网络
"""

import torch
from torch import nn

import torch.nn.functional as F


class Action_Net_CNN(nn.Module):
    def __init__(self):
        super(Action_Net_CNN, self).__init__()
        self.row = 40
        self.col = 42  # 6*7
        self.cov_depth = [4, 8]  # 2个卷积层的厚度

        # input data size = 1* row*col
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=1, padding=2),  # 4*40*42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # 4*(40/2)*(42/2) = 4*20*21
        )
        # input data size =  4*20*21
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 5, stride=1, padding=2),  # 8*20*21
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 3))  # 8*10*7 = 560
        )

        # 第一个线性层
        self.layer1 = nn.Linear(560, 1020)
        # Relu层
        self.layer2 = nn.ReLU()
        # 分类线性层
        self.layer3 = nn.Linear(1020, 200)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(200, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = self.conv2(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.dropout2d(x, p=0.5, training=self.training)
        x = self.layer4(x)
        x = self.layer5(x)

        out = F.log_softmax(x, dim=1)
        return out

if __name__ == '__main__':
    cnn = Action_Net_CNN()
    print(cnn)
