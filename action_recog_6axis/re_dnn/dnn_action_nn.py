# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/10/28 9:23
@Describe：

"""

import torch
from torch import nn
import torch.nn.functional as F
from dnn_base_var import action_window_col, action_window_row

input_size = action_window_row * action_window_col
hidden_size = 100
output_size = 5


class Action_dnn(nn.Module):
    def __init__(self):
        super(Action_dnn, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(hidden_size, output_size)
        self.layer4 = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class Action_dnn_regular(nn.Module):
    def __init__(self):
        super(Action_dnn_regular, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 500)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(500, hidden_size)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Linear(hidden_size, output_size)
        self.layer6 = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        out = self.layer1(input)
        out = F.dropout2d(out, p=0.5, training=self.training)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.dropout2d(out, p=0.5, training=self.training)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


"""
class Action_cnn_regular(nn.Module):
    def __init__(self):
        super(Action_cnn_regular, self).__init__()

        # model structur:
        # 输入：分为7个通道，每个通道40*9（原始节点数据）=2520
        # 输出：分为21个通道,2520/21=120
        # 输入图像的通道数=10(灰度图像),输出通道
        # 卷积核的shape是3乘3的,扫描步长为1,不加padding
        self.conv1 = nn.Conv2d(7, 21, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(21, 63, kernel_size=(3, 3), stride=(1, 1))

        # with map pool: output = 63 * (9,9) feature-maps -> flatten

        self.fc1 = nn.Linear(504, 500)
        self.fc2 = nn.Linear(500, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # conv layers
        x = F.relu(self.conv1(x))  # shape: 1, 10, 46, 46
        x = F.max_pool2d(x, 2, 2)  # shape: 1, 10, 23, 23
        x = F.relu(self.conv2(x))  # shape: 1, 20, 19, 19
        x = F.max_pool2d(x, 2, 2)  # shape: 1, 20, 9, 9

        # flatten to dense layer:
        x = x.view(-1, 504)

        # dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output
"""
