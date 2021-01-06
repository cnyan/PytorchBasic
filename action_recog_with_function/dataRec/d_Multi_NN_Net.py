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
import torch.nn.functional as F


class MyMultiConvNet(nn.Module):
    def __init__(self, axis):
        super(MyMultiConvNet, self).__init__()

        self.conv1_layer = nn.Sequential(
            nn.Conv1d(in_channels=7 * axis,
                      out_channels=128,
                      kernel_size=5,
                      stride=1,
                      padding=2),  # 输入63*36，自上向下扫描
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.conv2_layer = nn.Sequential(
            nn.Conv1d(128, 128, 5, 1, 2),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, dilation=1)
        )
        self.conv3_layer = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            # nn.Dropout2d(p=0.5),
            nn.ReLU(),
        )
        self.conv4_layer = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, dilation=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * (36 // 2 // 2), 5),
            # nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.conv1_layer(x)
        x = self.conv2_layer(x)
        x = self.conv3_layer(x)
        x = self.conv4_layer(x)
        out = x.view(x.size(0), -1)
        output = self.classifier(out)
        return output


class MyMultiResCnnNet(nn.Module):
    def __init__(self, axis):
        super(MyMultiResCnnNet, self).__init__()
        self.res1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 5, 1, 2),
            nn.BatchNorm1d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.res2_layer = nn.Sequential(
            nn.Conv1d(128, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.res3_layer = nn.Sequential(
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.res4_layer = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1, 0),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1, 1, 0),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.res5_layer = nn.Sequential(
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.Dropout(),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 9, 5)
        )

    def forward(self, x):
        x = self.res1_layer(x)
        res = x
        x = self.res2_layer(x)
        x = x + res
        x = self.res3_layer(x)
        res = x
        x = self.res4_layer(x)
        x = x + res
        x = self.res5_layer(x)

        out = x.view(x.size(0), -1)
        output = self.classifier(out)
        return output
