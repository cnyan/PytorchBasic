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
import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataToTorch import ActionDataSets
from dataOtherTestToTorch import ActionTestDataSets
import AUtils
import time
from sklearn.decomposition import PCA
import joblib
import warnings

warnings.filterwarnings('ignore')


class CF_features():
    def __init__(self, modelNet, model_name, axis, data_category='other_test'):
        super(CF_features, self).__init__()
        self.model = modelNet
        self.axis = axis
        self.data_category = data_category

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{axis}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.model_name = model_name

        self.conv4layer_features = {}
        self.model.confluence4_layer.register_forward_hook(self.get_conv4layer_activation("confluence4_layer"))

        if data_category == 'other_test':
            # 提取实验室其他同学的cnn特征
            action_data_test_set = ActionTestDataSets(axis)
            self.test_action_data_set = DataLoader(action_data_test_set, shuffle=True, num_workers=2)
        else:
            # 提取运动员的、训练、测试特征
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

            cf_features_data = self.conv4layer_features["confluence4_layer"]  # 2304
            cf_features_data = np.append(cf_features_data, int(label))
            cf_features_set.append(cf_features_data)

        cf_features_set = np.array(cf_features_set)
        print(cf_features_set.shape)
        np.save(savePath, cf_features_set)

    def get_conv4layer_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv4layer_features[name] = output.detach().cpu().numpy().flatten()

        return hook

class Decomposition():
    def __init__(self):
        pass

    def decomposition(self, axis, data_category='other_test',n_components=0.90):
        '''
            对实验数据降维
        :param X:数据集
        :return:X_pca
        '''
        dataPath = f'src/ml_cf_features/{data_category}_features_mat-{axis}.npy'

        X = np.load(dataPath)
        print(f"降维前的参数:{data_category}_{axis},data shape:{X.shape}")

        savePath = f'src/ml_cf_features_pca/{data_category}_features_pca_mat-{axis}.npy'

        if data_category == 'train':
            de_model = PCA(n_components=n_components, svd_solver='auto', whiten=True)

            X_data = X[:, :-1]
            X_label = X[:, -1]

            de_model.fit(X_data)
            joblib.dump(de_model, f'src/ml_pca_cf_model/pca_model_{axis}.pkl')
            X_de = de_model.transform(X_data)
            X_de = DataFrame(X_de, columns=['pca' + str(i) for i in np.arange(len(X_de[0]))]).round(6)
            X_de['SPE'] = X_label
            print(X_de)

            # 它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分,
            # 返回各个成分各自的方差百分比(贡献率) = 0.95
            pca_feature = np.array(X_de)

            print(f"降维后的参数:{data_category}{axis},维度是{len(pca_feature[0]) - 1},data shape:{pca_feature.shape}")
            print('成分各自的方差百分比(贡献率):{}'.format(np.add.reduce(de_model.explained_variance_ratio_)))

            print(np.array(de_model.explained_variance_ratio_))
            np.save(savePath, pca_feature)
        else:
            pca_model = joblib.load(f'src/ml_pca_cf_model/pca_model_{axis}.pkl')

            test_data = X[:, :-1]
            test_label = X[:, -1]
            test_pca_feature = pca_model.transform(test_data)
            test_pca_feature = DataFrame(test_pca_feature,
                                         columns=['pca' + str(i) for i in np.arange(len(test_pca_feature[0]))]).round(6)
            test_pca_feature['SPE'] = test_label
            pca_feature = np.array(test_pca_feature)

            print(f"降维后的参数:{data_category}{axis},维度是{len(pca_feature[0]) - 1},data shape:{pca_feature.shape}")
            np.save(savePath, pca_feature)




if __name__ == '__main__':
    from fg_cnn_fine_grained import MyMultiTempSpaceConfluenceNet

    is_extract_cnn = True

    if is_extract_cnn:
        # 运动员数据集处理
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

        # 其他人的测试集
        for axis in ['9axis', '6axis']:
            myMultiTempSpaceConfluenceNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))
            models_all = {'myMultiTempSpaceConfluenceNet': myMultiTempSpaceConfluenceNet}

            for model_name, model in models_all.items():
                print('===================********begin begin begin*********=================')
                print(f'当前执行参数：model={model_name}_{axis}')

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                cf_features = CF_features(model, model_name, axis=axis, data_category='other_test')
                cf_features.predict()

    #  开始降维
    for data_category in ['train', 'test']:
        for axis in ['9axis', '6axis']:
            # 开始处理降维数据
            decom = Decomposition()
            decom.decomposition(axis, data_category)

    for axis in ['9axis', '6axis']:
        # 开始处理降维数据
        decom = Decomposition()
        decom.decomposition(axis, data_category='other_test')



