#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import joblib
import glob
import os


class read_csv:
    def __init__(self, file_path='D:\\temp\\action_windows-6axis'):
        self.file_path = file_path
        self.file_list = glob.glob(os.path.join(self.file_path, '*.csv'))
        self.names = ['ER', 'aAX', 'aAY', 'aAZ', 'aWX', 'aWY', 'aWZ', 'bAX', 'bAY', 'bAZ', 'bWX', 'bWY',
                      'bWZ', 'cAX', 'cAY', 'cAZ', 'cWX', 'cWY', 'cWZ', 'dAX', 'dAY', 'dAZ', 'dWX', 'dWY', 'dWZ',
                      'eAX', 'eAY', 'eAZ', 'eWX', 'eWY', 'eWZ', 'fAX', 'fAY', 'fAZ', 'fWX', 'fWY', 'fWZ',
                      'gAX', 'gAY', 'gAZ', 'gWX', 'gWY', 'gWZ', 'ACC']
        self.windows_len = 40
        self.dataMat = []

    def read(self):
        for file in self.file_list:
            df = pd.read_csv(file, names=self.names)
            df_acc = np.array(df[1:]['ACC'], dtype=float)
            df_acc = np.reshape(df_acc, (-1, self.windows_len))

            # 获取数据
            for data in df_acc:
                data_0 = data[0:20]  # 定义为0 动作起始阶段
                data_0_flag = np.append(data_0, 0)
                data_1 = data[20:40]  # 定义为1 动作加速阶段
                data_1_flag = np.append(data_1, 1)

                self.dataMat.append(data_0_flag)
                self.dataMat.append(data_1_flag)

        print(f'读取数据总长度是：{len(self.dataMat)}')
        na = np.array(self.dataMat)
        # print(na)
        np.save('src/actionData.npy', na)
        return self.dataMat

    def split_data(self):
        read_path = 'src/actionData.npy'
        if not os.path.exists(read_path):
            self.read()

        data = np.load(read_path)

        print(data.shape)
        data_f = np.reshape(data, (-1, 2, 21))  # 三维数组(-1,一个动作的长度40/20=2，20+1flag=21)
        from sklearn.model_selection import train_test_split

        X_t, X_v, Y_t, Y_v = train_test_split(data_f[:, :, :-1], data_f[:, :, -1:], test_size=0.3)

        X_t = np.reshape(X_t, (-1, 20))
        X_v = np.reshape(X_v, (-1, 20))
        Y_t = np.reshape(Y_t, (-1, 1))
        Y_v = np.reshape(Y_v, (-1, 1))

        train_data = []
        for i in range(0, len(X_t)):
            d = np.append(X_t[i], Y_t[i][0])
            train_data.append(d)
        train_data = np.array(train_data)

        valid_data = []
        for i in range(0, len(X_v)):
            d = np.append(X_v[i], Y_v[i][0])
            valid_data.append(d)
        valid_data = np.array(valid_data)

        np.random.shuffle(train_data)
        np.random.shuffle(valid_data)

        # 正则化数据
        scaler = preprocessing.Normalizer().fit(train_data)
        scaler_path = 'src/normalize.pkl'
        joblib.dump(scaler, scaler_path)
        print('正则化建模完毕')

        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_valid = valid_data[:, :-1]
        y_valid = valid_data[:, -1]

        # 数据正则化处理
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)

        print(f'训练集：{x_train.shape},验证集:{x_valid.shape}')
        y_train = y_train.reshape(-1, 1)  # 升高一个维度
        y_valid = y_valid.reshape(-1, 1)  # 升高一个维度
        return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    ACTION_ROOT_PATH = 'D:\\temp\\action_windows-6axis'
    re = read_csv(ACTION_ROOT_PATH)
    re.read()
    re.split_data()
