# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/24 22:13
@Describe：

"""
import torch
from re_cnn_data_input import Input_Data


class Cnn_Predict():
    def __init__(self):
        self.model = torch.load('src/re_cnn_model.pkl')
        self.model.eval()
        self.test_action_data_set = Input_Data('src/test_action_data.npy')
        print(f'test_data size:{len(self.test_action_data_set)}')
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    cnn_predict = Cnn_Predict()
