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

