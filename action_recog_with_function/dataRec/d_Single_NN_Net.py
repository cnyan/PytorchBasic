# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/30 20:24
@Describe：

"""
import torch.nn as nn


class MyDnnNet(nn.Module):
    def __init__(self, input_size):
        super(MyDnnNet, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 256),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        # x = x.detach().numpy().flatten()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MyConvNet(nn.Module):
    def __init__(self, axis):
        super(MyConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=7 * axis,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # 输入36*63，自上向下扫描
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (36 // 3), 512),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        out = x_1.view(x_1.size(0), -1)
        output = self.classifier(out)
        return output

class MyDilaConvNet(nn.Module):
    def __init__(self, axis):
        super(MyConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=7 * axis,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      dilation=2),  # 输入36*63，自上向下扫描
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (36 // 3), 512),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        out = x_1.view(x_1.size(0), -1)
        output = self.classifier(out)
        return output
