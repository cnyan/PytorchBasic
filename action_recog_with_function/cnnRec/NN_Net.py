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
from torchvision import models


class MyDnn(nn.Module):
    def __init__(self, input_size):
        super(MyDnn, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        # x = x.detach().numpy().flatten()
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MyConvNet(nn.Module):
    def __init__(self, in_chanels, inputsize):
        super(MyConvNet, self).__init__()
        # width = (int(int(inputsize[0] // 3) / 2))
        # height = (int(int(inputsize[1] // 3) / 2))
        width = int(inputsize[0] / 3)
        height = int(inputsize[1] / 3)
        # print(width,height)
        # 输入 3*36*21
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chanels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool2d(3, 3)
        )  # （32，12，7）

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # (64,6,3)

        self.classifier = nn.Sequential(
            nn.Linear(32 * width * height, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(256, 5)
        )
        # self.initialize_weights()

    def forward(self, x):
        x_1 = self.conv1(x)
        # print(x_1.shape)
        # x_2 = self.conv2(x_1)
        # print(x_2.shape)
        out = x_1.view(x_1.size(0), -1)
        output = self.classifier(out)
        return output

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight.data)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight.data, 0, 1)
    #             m.bias.data.zero_()


class MyDilConvNet(nn.Module):
    def __init__(self, in_chanels, inputsize):
        super(MyDilConvNet, self).__init__()
        dilation = 1
        width = int(inputsize[0] / 3)
        height = int(inputsize[1] / 3)
        # print(width,height)
        # 输入 3*36*21
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chanels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1, dilation=dilation),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool2d(3, 3)
        )  # （32，12，7）

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 1, 1, dilation=dilation),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )  # (64,6,3)

        self.classifier = nn.Sequential(
            nn.Linear(32 * width * height, 512),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(256, 5)
        )
        # self.initialize_weights()

    def forward(self, x):
        x_1 = self.conv1(x)
        # print(x_1.shape)
        # x_2 = self.conv2(x_1)
        # print(x_2.shape)
        out = x_1.view(x_1.size(0), -1)
        output = self.classifier(out)
        return output


class MyVgg16Net(nn.Module):
    def __init__(self):
        super(MyVgg16Net, self).__init__()
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
    myConvnet = MyConvNet(3, (36, 21))
    myDnn = MyDnn(3 * 36 * 21)
    # 模型可视化
    # from torchviz import make_dot
    # x = torch.randn(1, 3, 36, 21).requires_grad_(True)
    # y = myConvnet(x)
    # myConvnet_dot = make_dot(y, params=dict(list(myConvnet.named_parameters()) + [('x', x)]))
    # myConvnet_dot.format = 'png'
    # myConvnet_dot.directory = r'src/model_img/'
    # myConvnet_dot.view()

    # import hiddenlayer as hl
    #
    # my_hl = hl.build_graph(myDnn, torch.zeros([1, 3, 36, 21]))
    # my_hl.theme = hl.graph.THEMES['blue'].copy()
    # my_hl.save('src/model_img/my_DNN.png', format='png')
    myvgg = MyVgg16Net()
    print(myvgg)
