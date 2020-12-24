# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/23 19:26
@Describe：

"""
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MyDnn(nn.Module):
    def __init__(self, input_size):
        super(MyDnn, self).__init__()
        self.layer1 = nn.Linear(input_size, 512)
        self.layer2 = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(256, 128)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(128, 5)

    def forward(self, x):
        # x = x.detach().numpy().flatten()
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x


class MyConvNet(nn.Module):
    def __init__(self, in_chanels, inputsize):
        super(MyConvNet, self).__init__()
        width = (math.ceil(math.ceil(inputsize[0] // 3) // 2))
        height = (math.ceil(math.ceil(inputsize[1] // 3) / 2))
        # 输入 3*36*21
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chanels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, 3)
        )  # （32，12，7）

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )  # （32，12，7）

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # (64,6,3)

        self.classifier = nn.Sequential(
            nn.Linear(64 * width * height, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)
        out = x_3.view(x_3.size(0), -1)
        output = self.classifier(out)
        return output


class MyDilConvNet(nn.Module):
    def __init__(self, in_chanels, inputsize):
        super(MyDilConvNet, self).__init__()
        width = int((math.ceil(math.ceil((inputsize[0]-2) // 3)-2) // 2))
        height = int((math.ceil(math.ceil((inputsize[1]-2) // 3)-2) / 2))
        # print(width,height)
        # 输入 3*36*21
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chanels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,dilation=2),
            nn.ReLU(),
            nn.AvgPool2d(3, 3)
        )  # （32，12，7）

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1,dilation=2),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )  # （32，12，7）

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 1, 1,dilation=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # (64,6,3)

        self.classifier = nn.Sequential(
            nn.Linear(64 * width * height, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x_1 = self.conv1(x)
         #print(x_1.shape)
        x_2 = self.conv2(x_1)
        # print(x_2.shape)
        x_3 = self.conv3(x_2)
        # print(x_3.shape)
        out = x_3.view(x_3.size(0), -1)
        output = self.classifier(out)
        return output

if __name__ == '__main__':
    myConvnet = MyConvNet(3)
    myDnn = MyDnn()
    # 模型可视化
    # from torchviz import make_dot
    # x = torch.randn(1, 3, 36, 21).requires_grad_(True)
    # y = myConvnet(x)
    # myConvnet_dot = make_dot(y, params=dict(list(myConvnet.named_parameters()) + [('x', x)]))
    # myConvnet_dot.format = 'png'
    # myConvnet_dot.directory = r'src/model_img/'
    # myConvnet_dot.view()

    import hiddenlayer as hl

    my_hl = hl.build_graph(myDnn, torch.zeros([1, 3, 36, 21]))
    my_hl.theme = hl.graph.THEMES['blue'].copy()
    my_hl.save('src/model_img/my_DNN.png', format='png')
