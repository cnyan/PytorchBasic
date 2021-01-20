# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/20 22:49
@Describe：
    获取测试数据的网络特征
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataToTorch import ActionDataSets
import AUtils
import time
import warnings

warnings.filterwarnings('ignore')


class CF_features():
    def __init__(self, modelNet, model_name, axis, data_category):
        super(CF_features, self).__init__()
        self.model = modelNet
        self.axis = axis
        self.data_category = data_category

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{axis}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.model_name = model_name

        action_data_test_set = ActionDataSets(data_category, axis)

        self.test_action_data_set = DataLoader(action_data_test_set, shuffle=True, num_workers=2)
        print(f'{self.data_category}_data shape: ({len(action_data_test_set)}{(action_data_test_set.data_shape())})')

    def predict(self):
        cf_features_set = []
        savePath = f'src/ml_cf_features/{self.data_category}_features_mat-{self.axis}.npy'

        for data, label in self.test_action_data_set:
            # data = data.unsqueeze(0)  # 扩展一个维度
            label = torch.LongTensor([int(label)])
            if torch.cuda.is_available():
                data = data.cuda()

            output = self.model(data)
            cf_features_data = output.cpu().data.numpy()[0]
            cf_features_data = np.append(cf_features_data,int(label))

            cf_features_set.append(cf_features_data)

        cf_features_set = np.array(cf_features_set)
        print(cf_features_set.shape)
        np.save(savePath, cf_features_set)


if __name__ == '__main__':
    from d_Multi_NN_Net import MyMultiTempSpaceConfluenceNet, MyMultiConvConfluenceNet

    for data_category in ['train', 'test']:
        for axis in ['9axis', '6axis']:

            myMultiTempSpaceConfluenceNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))

            models_all = {'myMultiTempSpaceConfluenceNet': myMultiTempSpaceConfluenceNet}

            for model_name, model in models_all.items():
                print('===================********begin begin begin*********=================')
                print(f'当前执行参数：model={model_name}_{axis}')

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                cf_features = CF_features(model, model_name, axis=axis, data_category=data_category)
                cf_features.predict()
