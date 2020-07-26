# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/26 20:49
@Describe：

"""

import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import platform
import torch
from glob import glob

action_window_row = 40
action_window_col = 6


def save_data(data_model='train'):
    print(data_model + '数据开始处理')
    action_root_path = "../action_recog_6axis/re_cnn/src/" + data_model + '_action_data.npy'

    df_array = np.load(action_root_path)
    df_data = df_array[:, :-1]
    df_labels = df_array[:, -1:]

    all_node_1 = []
    all_node_10 = []
    all_node_2 = []
    all_node_3 = []
    all_node_4 = []
    all_node_6 = []
    all_node_7 = []
    for i, data in enumerate(df_data):
        data = data.reshape(40, 42)

        node_1 = data[:, 0:6].flatten()
        node_1 = np.append(node_1, df_labels[i][0]).tolist()
        all_node_1.append(node_1)

        node_10 = data[:, 6:12].flatten()
        node_10 = np.append(node_10, df_labels[i][0]).tolist()
        all_node_10.append(node_10)

        node_2 = data[:, 12:18].flatten()
        node_2 = np.append(node_2, df_labels[i][0]).tolist()
        all_node_2.append(node_2)

        node_3 = data[:, 18:24].flatten()
        node_3 = np.append(node_3, df_labels[i][0]).tolist()
        all_node_3.append(node_3)

        node_4 = data[:, 24:30].flatten()
        node_4 = np.append(node_4, df_labels[i][0]).tolist()
        all_node_4.append(node_4)

        node_6 = data[:, 30:36].flatten()
        node_6 = np.append(node_6, df_labels[i][0]).tolist()
        all_node_6.append(node_6)

        node_7 = data[:, 36:42].flatten()
        node_7 = np.append(node_7, df_labels[i][0]).tolist()
        all_node_7.append(node_7)

    all_node_1 = np.array(all_node_1)
    all_node_10 = np.array(all_node_10)
    all_node_2 = np.array(all_node_2)
    all_node_3 = np.array(all_node_3)
    all_node_4 = np.array(all_node_4)
    all_node_6 = np.array(all_node_6)
    all_node_7 = np.array(all_node_7)

    print(all_node_1.shape)
    print(all_node_10.shape)
    print(all_node_2.shape)
    print(all_node_3.shape)
    print(all_node_4.shape)
    print(all_node_6.shape)
    print(all_node_7.shape)

    for file in glob(os.path.join('src/nodeData/' + data_model, '*.npy')):
        os.remove(file)

    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_1.npy', all_node_1)
    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_10.npy', all_node_10)
    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_2.npy', all_node_2)
    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_3.npy', all_node_3)
    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_4.npy', all_node_4)
    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_6.npy', all_node_6)
    np.save('src/nodeData/' + data_model + '/' + data_model + '_action_data_7.npy', all_node_7)


class Input_Data(Dataset):
    def __init__(self, data_file_path):
        """
        初始化
        :param data_file_name:  数据集路径,用于读取是验证机、训练集、测试集
        :param action_root_path:
        """
        super(Input_Data, self).__init__()
        df_array = np.load(data_file_path)
        df_data = df_array[:, :-1]
        df_labels = df_array[:, -1:]

        self.data = []
        self.labels = []
        self.toTensor = transforms.ToTensor()

        for i in range(len(df_array)):
            self.data.append(df_data[i].reshape(action_window_row, action_window_col))
            self.labels.append(int(df_labels[i]))

    def __getitem__(self, item):
        data = self.data[item]
        label = self.labels[item]
        data = self.feature_normalize(data)
        data = self.toTensor(data)
        data = data.to(torch.float32)
        # print(data.shape)
        return data, label

    def __len__(self):
        return len(self.data)

    # 矩阵归一化
    def feature_normalize(self, df):
        _range = np.max(df) - np.min(df)
        df = (df - np.min(df)) / _range
        return df


if __name__ == '__main__':
    # # 处理数据集合
    # for data_model in ['train', 'valid', 'test']:
    #     save_data(data_model)

    input_data = Input_Data("src/nodeData/train/train_action_data_1.npy")
    print(input_data[0])

