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
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # 输入36*63，自上向下扫描
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (36 // 3), 512),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        # print(x_1.shape)
        out = x_1.view(x_1.size(0), -1)
        output = self.classifier(out)
        return output


class MyDilaConvNet(nn.Module):
    def __init__(self, axis):
        super(MyDilaConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=7 * axis,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2),  # 输入36*63，自上向下扫描
            nn.Dropout2d(p=0.5),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * (36 // 3), 512),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x_1 = self.conv1(x)
        out = x_1.view(x_1.size(0), -1)
        output = self.classifier(out)
        return output


class MyLstmNet(nn.Module):
    def __init__(self, axis):
        super(MyLstmNet, self).__init__()
        self.axis = axis

        self.lstm = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=7 * axis,  # 图片每行的数据像素点
            hidden_size=7 * axis*2,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            # dropout=0.5,
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            bidirectional=False,  # 单向LSTM
        )

        self.out = nn.Linear(7 * axis*2, 5)  # 输出层

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        x = x.view(-1, 36, 7 * self.axis)
        r_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全0的 state
        # r_out = F.relu(r_out[:, -1, :])
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out
