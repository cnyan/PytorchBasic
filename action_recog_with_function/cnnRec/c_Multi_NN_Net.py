# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/28 17:21
@Describe：

"""

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models


class Multi_MyConvNet(nn.Module):
    def __init__(self, width_height_axis_pools):
        super(Multi_MyConvNet, self).__init__()
        # 输入 3*36*21
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=3)
        )  # （32，12，7）

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # (64,6,3)

        self.classifier = nn.Sequential(
            nn.Linear(int(
                64 * width_height_axis_pools['width'] * width_height_axis_pools['height']), 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Linear(128, 5)
        )
        # self.initialize_weights()

    def forward(self, x):
        x_1 = self.conv1(x)
        # print(x_1.shape)
        x_2 = self.conv2(x_1)
        # print(x_2.shape)
        out = x_2.view(x_2.size(0), -1)
        output = self.classifier(out)
        return output


class Multi_MyVgg16Net(nn.Module):
    def __init__(self, width_height_axis_pools):
        super(Multi_MyVgg16Net, self).__init__()
        # 预训练的vgg特征提取层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            # nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            # nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            # nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            # nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)
        )

        # 新的全连接层次
        self.classifier = nn.Sequential(
            nn.Linear(128 * width_height_axis_pools['width'] * width_height_axis_pools['height'], 2048),
            nn.ReLU(),
            # nn.Dropout2d(p=0.5),
            nn.Linear(2048, 512),
            # nn.Dropout2d(p=0.5),
            nn.Linear(512, 5)

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


if __name__ == '__main__':
    vgg16 = models.vgg16(pretrained=False)
    print(vgg16)
    multi_myvgg = Multi_MyVgg16Net()
    print(multi_myvgg)
