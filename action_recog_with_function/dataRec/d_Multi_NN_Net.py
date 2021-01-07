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
    """
    多卷积网络
    """

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
    """
    多卷积残差网络
    """

    def __init__(self, axis):
        super(MyMultiResCnnNet, self).__init__()

        self.res1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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
            nn.Dropout(0.5),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

        )

        self.res3_layer = nn.Sequential(
            nn.Conv1d(128, 256, 1, 2, 0),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        self.res4_layer = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1, 0),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1, 1, 0),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),

        )

        self.res5_layer = nn.Sequential(
            nn.Conv1d(256, 256, 1, 1, 0),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * (36 // 2 // 2), 5)
        )

    def forward(self, x):
        out = self.res1_layer(x)
        res = out
        out = self.res2_layer(out)
        out = out + res
        out = self.res3_layer(out)
        res = out
        out = self.res4_layer(out)
        out = out + res
        out = self.res5_layer(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        output = self.classifier(out)
        return output


class MyMultiConvLstmNet(nn.Module):
    """
    卷积+LSTM
    """

    def __init__(self, axis):
        super(MyMultiConvLstmNet, self).__init__()
        self.conv1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 3, 1, 1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # self.embedding = nn.Embedding(36,7*axis)

        self.lstm_layer = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=18,  # 图片每行的数据像素点
            hidden_size=128,  # rnn hidden unit
            num_layers=2,  # 有几层 RNN layers
            dropout=0.5,
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            bidirectional=False,  # 单向LSTM
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv1_layer(x)
        r_out, (h_n, h_c) = self.lstm_layer(x, None)
        # print(r_out[:, -1, :].shape)
        out = self.classifier(r_out[:, -1, :])
        return out


class MyMultiConvConfluence(nn.Module):
    """
    多层卷积融合
    """

    def __init__(self, axis):
        super(MyMultiConvConfluence, self).__init__()
        self.conv1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )
        self.temporal2_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, 2, 2),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18
        self.spatial3_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18

        self.confluence4_layer = nn.Sequential(
            nn.Conv1d(128, 256, 2, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )  # 256*9

        self.classifier = nn.Sequential(
            nn.Linear(256 * 9, 5)
        )

    def forward(self, x):
        conv1 = self.conv1_layer(x)
        temp = self.temporal2_layer(x)
        spital = self.spatial3_layer(x)
        out = self.confluence4_layer(conv1 + temp + spital)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class MyIncepConvNet(nn.Module):
    """
    inception 网络
    """

    def __init__(self):
        super(MyIncepConvNet, self).__init__()


if __name__ == '__main__':
    # 模型可视化
    axis = '9axis'
    myMultiConvNet = MyMultiConvNet(int(axis[0]))
    myMultiResCnnNet = MyMultiResCnnNet(int(axis[0]))
    myMultiConvLstmNet = MyMultiConvLstmNet(int(axis[0]))

    import torch
    import os
    import shutil
    from torchviz import make_dot

    x = torch.randn(1, 63, 36).requires_grad_(True)
    y = myMultiResCnnNet(x)
    myConvnet_dot = make_dot(y, params=dict(list(myMultiResCnnNet.named_parameters()) + [('x', x)]))
    myConvnet_dot.format = 'png'
    myConvnet_dot.directory = r'src/model_img/myMultiResCnnNet'
    myConvnet_dot.view()

    import hiddenlayer as hl

    my_hl = hl.build_graph(myMultiResCnnNet, torch.zeros([1, 63, 36]))
    my_hl.theme = hl.graph.THEMES['blue'].copy()
    my_hl.save(r'src/model_img/myMultiResCnnNet_hl.png', format='png')

    # import netron
    #
    # modelPath = "src/model/myLstmNet_9axis_model.pkl"
    # netron.start(modelPath)
