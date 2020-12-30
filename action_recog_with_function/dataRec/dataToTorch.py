# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/30 15:53
@Describe：

"""
import platform
from glob import glob
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch

np.set_printoptions(suppress=True)


class DataToTorch():
    """
    将窗口数据转为训练集和数据集、测试集
    """

    def __init__(self, windowDataFoldPath, axis):
        super().__init__()
        self.internalNodeNum = 7
        self.action_window_row = 36  # 窗口长度
        self.num_of_windows = 2000  # 每个动作提取多少
        self.axis = axis
        self.action_window_col = self.internalNodeNum * int(axis[1])
        self.windowDataFoldPath = windowDataFoldPath
        self.data_torch_path = f'src/torchData/trainingData/'

    def readWindowsToTorchData(self, test_size=0.1, valid_size=0.1):
        files_list = glob(os.path.join(self.windowDataFoldPath, '*.csv'))
        test_size = int(self.num_of_windows * test_size)
        valid_size = int(self.num_of_windows * 5 * valid_size)

        all_torch_mat = []  # 总的数据集
        test_torch_mat = []
        for file_name in files_list:
            print(file_name)
            label = int(file_name[-5])
            data_mat = pd.read_csv(file_name, dtype=float, header=0).round(3)
            data_mat = np.array(data_mat)[:, 1:-1]

            data_mat = data_mat[:int(len(data_mat) / self.action_window_row) * self.action_window_row, :]  # 确保是窗口长度的倍数
            data_mat = np.reshape(data_mat, (-1, int(self.action_window_row), int(self.action_window_col)))
            # print(data_mat.shape)
            data_mat = data_mat[:self.num_of_windows, :, ]  # 每个动作取2000个
            # print(data_mat.shape)
            for i, df in enumerate(data_mat):
                df = df.flatten()
                df = np.append(df, label)
                if i > test_size - 1:
                    all_torch_mat.append(df)
                else:
                    test_torch_mat.append(df)

        all_torch_mat = np.array(all_torch_mat)
        test_torch_mat = np.array(test_torch_mat)
        all_torch_mat = np.random.permutation(all_torch_mat)
        valid_torch_mat = all_torch_mat[:valid_size, :]
        train_torch_mat = all_torch_mat[valid_size:, :]

        np.save(str(os.path.join(self.data_torch_path, f'test/test_torch_data{self.axis}.npy')), test_torch_mat)
        np.save(str(os.path.join(self.data_torch_path, f'valid/valid_torch_mat{self.axis}.npy')), valid_torch_mat)
        np.save(str(os.path.join(self.data_torch_path, f'train/train_torch_mat{self.axis}.npy')), train_torch_mat)


class ActionDataSets(Dataset):
    """
    封装数据集为DataSet
    """

    def __init__(self, torch_data_path):
        super(ActionDataSets, self).__init__()
        self.torch_data = np.load(torch_data_path)
        self.axis = int(torch_data_path[-9])
        self.standScaler = StandardScaler(with_mean=True, with_std=True)

    def __len__(self):
        return len(self.torch_data)

    def __getitem__(self, item):
        label = self.torch_data[item][-1]
        data = self.torch_data[item][:-1]
        data = data.reshape(-1, 7 * self.axis).T
        data = self.standScaler.fit_transform(data)
        return data, int(label)


if __name__ == '__main__':
    axiss = ['-6axis', '-9axis']  # 36,42  36,63
    for axis in axiss:
        if platform.system() == 'Windows':
            action_root_path = f'D:/home/DataRec/action_windows{axis}'
        else:
            action_root_path = f'/home/yanjilong/dataSets/DataRec/action_windows{axis}'

        dataToTorch = DataToTorch(action_root_path, axis)
        dataToTorch.readWindowsToTorchData()

    train_data_path = 'src/torchData/trainingData/train/train_torch_mat-9axis.npy'
    valid_data_path = 'src/torchData/trainingData/valid/valid_torch_mat-9axis.npy'
    test_data_path = 'src/torchData/trainingData/test/test_torch_mat-9axis.npy'
    # 读取数据
    train_action_data_sets = ActionDataSets(train_data_path)
    train_action_data_loader = DataLoader(train_action_data_sets,batch_size=64,shuffle=True,num_workers=2)

    for  batch_data in train_action_data_loader:
        inputs, labels = batch_data
        # label = torch.LongTensor([int(labels[0])])
        # print(labels)
        # print(labels.data)

