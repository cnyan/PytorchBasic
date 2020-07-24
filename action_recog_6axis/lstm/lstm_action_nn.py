# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/12/21 12:17
@Describe：

"""
import numpy as np
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt
import platform

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from lstm_base_var import action_window_row, action_window_col


# 行长：action_window_col  列宽：action_window_row
class ActionsDataSet(Dataset):

    def __init__(self, img_data_root, size=(action_window_col, action_window_row)):
        """
        :param img_data_path:
        :param size:  width = 63,height = 30
        """
        super().__init__()
        self.files = glob(img_data_root + '*.jpg')
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        # L = Image.open(self.files[item])
        # print(np.array(L).shape)
        # L = L.convert('L')
        # print(np.array(L).shape)

        img_data = np.asarray(Image.open(self.files[item]).resize(self.size))  # (40, 63, 3)

        if platform.system() == 'Windows':
            img_lable = self.files[item].split('\\')[-1].split('_')[0]
        else:
            img_lable = self.files[item].split('/')[-1].split('_')[0]
        img_name = self.files[item]

        L_image = np.array(Image.fromarray(img_data, mode='RGB').convert('L'))  # 灰度图像（40，63）
        L_image = np.clip(L_image / 255.0, 0, 1) # float64
        # L_image = L_image.astype(np.float16)
        return L_image, int(img_lable) - 1, img_name

    def show_img(self, item):
        """
        show image with item
        :param item:
        :return:
        """
        img_data, lab, name = self.__getitem__(item)
        img_data = img_data.transpose(1, 2, 0)  # transpose 按照指定维度旋转矩阵
        img_data = self.std * img_data + self.mean  # 反序列化
        img_data = np.clip(img_data, -1, 1)
        plt.imshow(img_data)
        plt.show()


# 自定义的神经网络
class Action_Net_LSTM(nn.Module):
    def __init__(self):
        super(Action_Net_LSTM, self).__init__()
        self.input_size = action_window_col
        self.hidden_size = 63
        self.lstm = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=self.input_size,
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            dropout=0.5,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        # self.dropout = nn.Dropout2d(0.2)
        self.out = nn.Linear(self.hidden_size, 5)


    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # r_out, _ = self.lstm(x, None)  # None represents zero initial hidden state
        # s, b, h = r_out.size()
        # r_out = x.view(s * b, h)
        # # choose r_out at the last time step
        # out = self.out(r_out)
        # out = out.view(s, b, -1)
        # return out
        # x = self.embedding(x)

        r_out, (h_n, h_c) = self.lstm(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        # out = self.dropout(r_out[:, -1, :])
        out = self.out(r_out[:, -1, :])
        return out
