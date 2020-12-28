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


class Multi_MyVgg16Net(nn.Module):
    def __init__(self):
        super(Multi_MyVgg16Net, self).__init__()
        # 预训练的vgg特征提取层
        vgg16 = models.vgg16(pretrained=True)
        vgg16 = vgg16.features
        for param in vgg16.parameters():
            param.requires_grad_(False)
        self.vgg16 = vgg16
        # 新的全连接层次
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output



if __name__ == '__main__':

    myvgg = Multi_MyVgg16Net()
    print(myvgg)
