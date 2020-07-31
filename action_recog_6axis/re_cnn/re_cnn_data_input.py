# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/24 15:42
@Describe：
    数据输入
"""
from glob import glob
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import platform
import torch

action_window_row = 40
action_window_col = 42


def save_data_sets(action_root_path=None, valid_size=0.20, test_size=0.05):
    """
    读取原始数据，并分为train、valid、test数据集
    :param action_root_path: 原始文件路径
    :return:
    """
    if action_root_path == None:
        if platform.system() == 'Windows':
            action_root_path = 'D:/temp/action_windows-6axis'
        else:
            action_root_path = '/home/yanjilong/DataSets/action_windows-6axis'

    # 导入数据 8 种动作，每种动作由7个节点数据构成
    file_list = []
    for maindir, subdir, file_name_list in os.walk(action_root_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            file_list.append(apath)
    # print(file_list)

    # action_array = np.zeros(shape=(0, 1900))
    action_array = np.zeros(shape=(0, (action_window_row * action_window_col) + 1))
    for filename in file_list:
        datamat = []
        labels = []
        print(filename)
        lab = int(filename.split('_')[-1].split('.')[0]) - 1
        label = [lab]
        # label = [0, 0, 0, 0, 0]
        # label[lab] = 1

        df = np.array(pd.read_csv(filename, dtype=float)).round(6)[:, 1:-1]

        df = df[:int(len(df) / action_window_row) * action_window_row, :]

        data = np.reshape(df, (-1, action_window_row, action_window_col))[:900, :]
        # print(data.shape)
        for i in range(len(data)):
            datamat.append(data[i].flatten())
            labels.append(label)
        print(np.array(datamat).shape)  # (行, 40*42)

        # 横向组合数据与标签
        complex_array = np.concatenate([np.array(datamat), np.array(labels)], axis=1)
        # print(complex_array.shape)
        # 纵向累加数据
        action_array = np.concatenate([action_array, complex_array], axis=0)

    # 随机排序
    np.random.shuffle(action_array)
    print(f'all_data size:{action_array.shape}')    # (4500, 1681)

    # 获得测试数据
    split_num = -int(len(action_array) * test_size)
    test_action_data = action_array[split_num:]
    action_array = action_array[:split_num]
    print(f'test_data size:{test_action_data.shape}') # (225, 1681)
    if os.path.exists('src/test_action_data.npy'):
        os.remove('src/test_action_data.npy')
    np.save('src/test_action_data.npy', test_action_data)

    split_num = -int(len(action_array) * valid_size)
    valid_action_data = action_array[split_num:]
    train_action_array = action_array[:split_num]
    print(f'train_data size:{train_action_array.shape}') # (3420, 1681)
    print(f'valid_data size:{valid_action_data.shape}') #(855, 1681)

    if os.path.exists('src/train_action_data.npy'):
        os.remove('src/train_action_data.npy')
    np.save('src/train_action_data.npy', train_action_array)

    if os.path.exists('src/valid_action_data.npy'):
        os.remove('src/valid_action_data.npy')
    np.save('src/valid_action_data.npy', valid_action_data)


class Input_Data(Dataset):
    def __init__(self, data_file_path, action_root_path=None):
        """
        初始化
        :param data_file_name:  数据集路径,用于读取是验证机、训练集、测试集
        :param action_root_path:
        """
        super(Input_Data, self).__init__()

        df_array = np.load('src/' + data_file_path)
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
        _range = np.max(df, axis=0) - np.min(df, axis=0)
        df = (df - np.min(df, axis=0)) / _range
        df = np.nan_to_num(df)
        return df


if __name__ == '__main__':
    save_data_sets()

    input_data = Input_Data("train_action_data.npy")
    print(input_data[0])
