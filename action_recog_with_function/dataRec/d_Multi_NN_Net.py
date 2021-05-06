# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/30 20:24
@Describe：

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MyMultiConvNet(nn.Module):
    """
    多卷积网络
    """

    def __init__(self, axis):
        super(MyMultiConvNet, self).__init__()

        self.conv0_layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 0),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout2d(0.5),
            nn.ReLU(),

            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 0),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

        self.globel_avgpool = nn.AdaptiveAvgPool2d((3, 3))  # (256,7*axis,3)
        # self.torch_sum = lambda x:torch.sum(x,dim=0)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 9, 5),
            # nn.Linear(512, 5)
        )

    def forward(self, x):
        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        out = self.conv0_layer(x_2d)

        out = self.globel_avgpool(out)

        out = out.view(out.size(0), -1)
        output = self.classifier(out)
        return output, 0


class MyMultiConvNet_2(nn.Module):
    """
    改进的Inception结构，加入一维卷积 MDFF-CNN
    """

    def __init__(self, axis):
        super(MyMultiConvNet_2, self).__init__()

        self.conv1_layer = nn.Sequential(
            MyInception_2d(1, 16, axis),
            # MyInception_2d(16 * 4, 32, axis),
        )  # (64,32 * 4 ,* 7*axis,36)

        self.confluence2_layer = nn.Sequential(
            nn.Conv2d(16* 4, 128, 3, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )  # 256*9

        self.globel_avgpool = nn.AdaptiveAvgPool2d((3,3))  # (256,7*axis,3)

        self.classifier = nn.Sequential(
            nn.Linear(256*9, 5),
        )

    def forward(self, x):
        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        out = self.conv1_layer(x_2d)

        mix_data = torch.sum(out, dim=1)  # 合并之后，只有三个维度
        out = self.confluence2_layer(out)
        out = self.globel_avgpool(out)

        out = out.view(out.size(0), -1)
        output = self.classifier(out)
        return output, mix_data


class MyInception_2d(nn.Module):
    def __init__(self, input_size, output_size, axis=0):
        super(MyInception_2d, self).__init__()
        conv_size = output_size*4
        # dim_size = (input_size // 4) if input_size > 4 else 4
        self.axis = axis
        self.out_size = output_size
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, output_size, 1, 1, 0),
            # nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout2d(),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_size, conv_size, 3, 1, 1),
            nn.BatchNorm2d(conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(conv_size, output_size, 1, 1, 0),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(36 * input_size, 36 * conv_size, 5, 1, 2),
            nn.BatchNorm1d(36 * conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Conv1d(36 * conv_size, 36 * conv_size, 3, 1, 1),
            # nn.BatchNorm1d(36 * conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Conv1d(36 * conv_size, 36 * output_size, 1, 1, 0),
            nn.BatchNorm1d(36 * output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(7 * axis * input_size, 7 * axis * conv_size, 5, 1, 2),
        #     nn.BatchNorm1d(7 * axis * conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     #nn.ReLU(),
        #     nn.Conv1d(7 * axis * conv_size, 7 * axis * output_size, 1, 1, 0),
        #     nn.BatchNorm1d(7 * axis * output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    # nn.ReLU(),
        # )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(input_size, output_size, 1, 1, 0),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_36 = x.permute([1, 0, 3, 2]).squeeze(0)  # 减少一个维度
        x3 = self.conv3(x_36)
        x3_size = x3.size()
        x3 = x3.view(x3_size[0], self.out_size, 7 * self.axis, 36)

        # x_axis = x.permute([1, 0, 2, 3]).squeeze(0)  # 减少一个维度
        # x5 = self.conv5(x_axis)
        # x5_size = x5.size()
        # x5 = x5.view(x5_size[0], self.out_size, 7 * self.axis, 36)

        x4 = self.conv4(x)
        outputs = [x1, x2, x3, x4]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class MyMultiConvNet_3(nn.Module):
    """
    纯Inception结构
    """

    def __init__(self, axis):
        super(MyMultiConvNet_3, self).__init__()

        self.conv1_layer = nn.Sequential(
            MyInception_3d(1, 16, axis),
        )  # (64,32 * 4 ,* 7*axis,36)

        self.conv2_layer = nn.Sequential(
            nn.Conv2d(16 * 4, 128, 2, 1, 1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )  # 256*9

        self.conv3_layer = nn.Sequential(
            nn.Conv2d(128, 256, 2, 1, 1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )  # 256*9
        self.globel_avgpool = nn.AdaptiveAvgPool2d((3, 3))  # (256,7*axis,3)

        self.classifier = nn.Sequential(
            nn.Linear(256 *9, 5),
        )

    def forward(self, x):
        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        out = self.conv1_layer(x_2d)
        out = self.conv2_layer(out)
        out = self.conv3_layer(out)
        out = self.globel_avgpool(out)

        out = out.view(out.size(0), -1)
        output = self.classifier(out)
        return output, 0


class MyInception_3d(nn.Module):
    def __init__(self, input_size, output_size, axis=0):
        super(MyInception_3d, self).__init__()
        conv_size = output_size * 4
        dim_size = (input_size // 4) if input_size > 4 else 4
        self.axis = axis
        self.out_size = output_size
        self.input_size = input_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_size, output_size, 1, 1, 0),
            # nn.Dropout2d(0.5),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_size, conv_size, 3, 1, 1),
            nn.BatchNorm2d(conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(conv_size, output_size, 1, 1, 0),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_size, conv_size, 5, 1, 2),
            nn.BatchNorm2d(conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.Conv2d(conv_size, conv_size, 3, 1, 1),
            # nn.BatchNorm2d(conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Conv2d(conv_size, output_size, 1, 1, 0),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(7 * axis * input_size, 7 * axis * dim_size, 1, 1, 0),
        #     nn.BatchNorm1d(7 * axis * dim_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Conv1d(7 * axis * dim_size, 7 * axis * conv_size, 5, 1, 2),
        #     nn.BatchNorm1d(7 * axis * conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Conv1d(7 * axis * conv_size, 7 * axis * output_size, 1, 1, 0),
        #     nn.BatchNorm1d(7 * axis * output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(input_size, output_size, 1, 1, 0),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        outputs = [x1, x2, x3, x4]
        outputs = torch.cat(outputs, dim=1)
        return outputs




class MyMultiConvNet_4(nn.Module):
    """
    纯Inception结构
    """

    def __init__(self, axis):
        super(MyMultiConvNet_4, self).__init__()

        self.conv1_layer = nn.Sequential(
            MyInception_4d(1, 4, axis),
        )  # (64,32 * 4 ,* 7*axis,36)

        self.conv2_layer = nn.Sequential(
            nn.Conv1d(4 * 4*36, 1024, 2, 1, 1),
            nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool1d(3, 3)
        )  # 256*9

        self.conv3_layer = nn.Sequential(
            nn.Conv1d(1024, 2048, 2, 1, 1),
            nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.AvgPool1d(3, 3)
        )  # 256*9
        self.globel_avgpool = nn.AdaptiveAvgPool1d((1))  # (256,7*axis,3)

        self.classifier = nn.Sequential(
            nn.Linear(2048 *1, 5),
        )

    def forward(self, x):
        x = x.permute([0,2,1])
        out = self.conv1_layer(x)
        out = self.conv2_layer(out)
        out = self.conv3_layer(out)
        out = self.globel_avgpool(out)

        out = out.view(out.size(0), -1)
        output = self.classifier(out)
        return output, 0


class MyInception_4d(nn.Module):
    def __init__(self, input_size, output_size, axis=0):
        super(MyInception_4d, self).__init__()
        self.conv_size = output_size * 2*36
        dim_size = (input_size // 4) if input_size > 4 else 4
        self.axis = axis
        self.output_size = output_size*36
        self.input_size = input_size*36

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.input_size, self.output_size, 1, 1, 0),
            # nn.Dropout2d(0.5),
            nn.BatchNorm1d(self.output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.input_size, self.conv_size, 3, 1,1),
            nn.BatchNorm1d(self.conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(self.conv_size, self.output_size, 1, 1, 0),
            nn.BatchNorm1d(self.output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(self.input_size, self.conv_size, 3, 1, 1),
            nn.BatchNorm1d(self.conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(self.conv_size,self.conv_size, 3, 1, 1),
            nn.BatchNorm1d(self.conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(self.conv_size, self.output_size, 1, 1, 0),
            nn.BatchNorm1d(self.output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(7 * axis * input_size, 7 * axis * dim_size, 1, 1, 0),
        #     nn.BatchNorm1d(7 * axis * dim_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Conv1d(7 * axis * dim_size, 7 * axis * conv_size, 5, 1, 2),
        #     nn.BatchNorm1d(7 * axis * conv_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     nn.Conv1d(7 * axis * conv_size, 7 * axis * output_size, 1, 1, 0),
        #     nn.BatchNorm1d(7 * axis * output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        # )
        self.conv4 = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
            nn.Conv1d(self.input_size, self.output_size, 1, 1, 0),
            nn.BatchNorm1d(self.output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        outputs = [x1, x2, x3, x4]
        outputs = torch.cat(outputs, dim=1)
        return outputs


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
        return output, 0


class MyMultiConvLstmNet(nn.Module):
    """
    卷积+LSTM
    """

    def __init__(self, axis):
        super(MyMultiConvLstmNet, self).__init__()

        self.conv1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )  # 128*18
        # self.embedding = nn.Embedding(36,7*axis)

        self.lstm_layer = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=18,  # 图片每行的数据像素点
            hidden_size=128,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
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
        return out, 0


class MyMultiConvConfluenceNet(nn.Module):
    """
    多层卷积融合
    """

    def __init__(self, axis):
        super(MyMultiConvConfluenceNet, self).__init__()
        self.conv1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )
        self.temporal2_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 1, 1, 0),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, 1, 2),
            nn.Dropout(0.5),
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
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.Dropout(0.5),
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
            nn.AvgPool1d(2, 2)
        )  # 256*9
        self.confluence5_layer = nn.Sequential(
            nn.Conv1d(256, 256, 2, 1, 1),
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
        out = self.confluence5_layer(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, 0


#
class MyMultiTempSpaceConfluenceNet(nn.Module):
    """
    时空卷积融合
    """

    def __init__(self, axis):
        super(MyMultiTempSpaceConfluenceNet, self).__init__()

        self.axis = axis
        self.temporal1_layer = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 2, dilation=2),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 36, 1, 1, 0),
            nn.BatchNorm1d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 输入大小（7*axis，36）
        self.spatial2_layer = nn.Sequential(
            nn.Conv2d(1, 128, 1, 1, 0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1, 1, 0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18

        self.confluence3_layer = nn.Sequential(
            nn.Conv2d(1, 128, 2, 1, 1),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )  # 256*9

        self.confluence4_layer = nn.Sequential(
            nn.Conv2d(128, 256, 2, 1, 1),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )  # 256*9

        self.classifier = nn.Sequential(
            nn.Linear(256 * (math.ceil((7 * axis) / 4)) * 9, 5),
        )

    def forward(self, x):
        x_1d = x.permute([0, 2, 1])
        temp = self.temporal1_layer(x_1d)
        temp = temp.unsqueeze(0).permute([1, 0, 3, 2])

        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        space = self.spatial2_layer(x_2d)

        input = temp + space  # [64, 1, 42, 36]
        # input_2d = input.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度

        out = self.confluence3_layer(input)
        out = self.confluence4_layer(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class MyMultiTestNet(nn.Module):
    """
    时空卷积融合
    """

    def __init__(self, axis):
        super(MyMultiTestNet, self).__init__()
        # self.conv1_layer = nn.Sequential(
        #     nn.Conv1d(7 * axis, 128, 1, 1, 0),
        #     nn.Dropout(0.5),
        #     nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.ReLU(),
        #     # nn.AvgPool1d(2, 2)
        # )
        self.axis = axis
        self.temporal1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 7 * axis, 1, 1, 0),
            nn.BatchNorm1d(7 * axis, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(7 * axis, 7 * axis, 5, 1, 2),
            nn.Dropout(0.5),
            nn.BatchNorm1d(7 * axis, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(7 * axis, 7 * axis, 1, 1, 0),
            nn.BatchNorm1d(7 * axis, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18
        self.spatial2_layer = nn.Sequential(
            nn.Conv2d(1, 1, 1, 1, 0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, 1, 0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18

        self.confluence3_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 2, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )  # 256*9

        self.confluence4_layer = nn.Sequential(
            nn.Conv1d(128, 256, 2, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )  # 256*9

        self.globel_avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Linear(256, 5)

    def forward(self, x):
        temp = self.temporal1_layer(x)

        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        spital = self.spatial2_layer(x_2d)
        spital = spital.permute([1, 0, 2, 3]).squeeze(0)

        out = self.confluence3_layer(temp + spital)
        out = self.confluence4_layer(out)
        out = self.globel_avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, 0


if __name__ == '__main__':
    # 模型可视化
    axis = '9axis'
    myMultiConvNet = MyMultiConvNet(int(axis[0]))
    myMultiConvNet_2 = MyMultiConvNet_2(int(axis[0]))
    myMultiConvNet_3 = MyMultiConvNet_3(int(axis[0]))
    myMultiResCnnNet = MyMultiResCnnNet(int(axis[0]))
    myMultiConvLstmNet = MyMultiConvLstmNet(int(axis[0]))
    myMultiConvConfluenceNet = MyMultiConvConfluenceNet(int(axis[0]))
    myMultiTempSpaceConfluenceNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))
    myMultiTestNet = MyMultiTestNet(int(axis[0]))

    import torch
    import os
    import shutil
    from torchviz import make_dot

    models_all = {'myMultiConvNet': myMultiConvNet, 'myMultiConvNet_2': myMultiConvNet_2,
                  'myMultiConvNet_3': myMultiConvNet_3}

    for model_name, model in models_all.items():
        x = torch.randn(1, 63, 36).requires_grad_(True)
        y, mixdata = model(x)
        myConvnet_dot = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
        myConvnet_dot.format = 'png'
        myConvnet_dot.directory = f'src/model_img/{model_name}'
        myConvnet_dot.view()

        # import hiddenlayer as hl
        #
        # my_hl = hl.build_graph(model, torch.zeros([1, 63, 36]))
        # my_hl.theme = hl.graph.THEMES['blue'].copy()
        # my_hl.save(fr'src/model_img/{model_name}_hl.png', format='png')

    # import netron
    #
    # modelPath = "src/model/myLstmNet_9axis_model.pkl"
    # netron.start(modelPath)
