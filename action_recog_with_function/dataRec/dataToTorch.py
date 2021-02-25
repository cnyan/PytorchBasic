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
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import joblib
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

np.set_printoptions(suppress=True)


class DataToTorch():
    """
    将窗口数据转为训练集和数据集、测试集,窗口正则化、提取时频特征
    """

    def __init__(self, windowDataFoldPath, axis):
        super().__init__()
        self.internalNodeNum = 7
        self.action_window_row = 36  # 窗口长度
        self.num_of_windows = 2000  # 每个动作提取多少
        self.axis = axis
        self.action_window_col = self.internalNodeNum * int(axis[0])
        self.windowDataFoldPath = windowDataFoldPath
        self.data_torch_path = f'src/torchData/trainingData/'

    def readWindowsToTorchData(self, test_size=0.1, valid_size=0.2):
        files_list = glob(os.path.join(self.windowDataFoldPath, '*.csv'))
        test_size = int(self.num_of_windows * test_size)
        valid_size = int(self.num_of_windows * 5 * valid_size)

        all_torch_mat = []  # 总的数据集
        test_torch_mat = []
        for file_name in files_list:
            print(file_name)
            label = int(file_name[-5]) - 1
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

        np.save(str(os.path.join(self.data_torch_path, f'test/test_torch_mat-{self.axis}.npy')), test_torch_mat)
        np.save(str(os.path.join(self.data_torch_path, f'valid/valid_torch_mat-{self.axis}.npy')), valid_torch_mat)
        np.save(str(os.path.join(self.data_torch_path, f'train/train_torch_mat-{self.axis}.npy')), train_torch_mat)

        print('{} train size:{}'.format(self.axis, train_torch_mat.shape))
        print('{} valid size:{}'.format(self.axis, valid_torch_mat.shape))
        print('{} test size:{}'.format(self.axis, test_torch_mat.shape))


class ActionDataSets(Dataset):
    """
    封装数据集为DataSet
    """

    def __init__(self, data_category='train', axis='9axis', torch_data_path=None):
        """
        初始化函数
        :param data_category: 数据集模式，有三种 train valid test
        :param axis: 几个轴的数据集，9axis 6axis
        :param torch_data_path: 默认数据集路径，如果为空则根据data_category和axis加载数据集
        """
        super(ActionDataSets, self).__init__()
        if torch_data_path == None:
            if data_category == 'train':
                torch_data_path = f'src/torchData/trainingData/train/train_torch_mat-{axis}.npy'
            elif data_category == 'valid':
                torch_data_path = f'src/torchData/trainingData/valid/valid_torch_mat-{axis}.npy'
            else:  # data_category == test
                torch_data_path = f'src/torchData/trainingData/test/test_torch_mat-{axis}.npy'
        else:
            torch_data_path = torch_data_path

        self.torch_data = np.load(torch_data_path)
        self.axis = int(axis[0])
        self.standScaler = StandardScaler(with_mean=True, with_std=True)
        # self.standScaler = MinMaxScaler()

    def __len__(self):
        return len(self.torch_data)

    def __getitem__(self, item):
        label = self.torch_data[item][-1]
        data = self.torch_data[item][:-1]
        data = data.reshape(-1, 7 * self.axis).T  # 转置，从[36,7*axis]转为[7*axis,36]
        data = self.standScaler.fit_transform(data)
        data = data.astype(np.float32)
        return data, int(label)

    def data_shape(self):
        data = self.torch_data[0][:-1]
        data = data.reshape(-1, 7 * self.axis).T

        return data.shape


class StandAndExtractTfFeatures():
    """
    # 从numpy数据中读取数据，分别保存正则化、特征提取之后的数据
    """

    def __init__(self, data_category='train', axis='9axis'):
        self.data_category = data_category
        self.axis = axis

    def saveStandScalerData(self):
        file_name = ''
        if self.data_category == 'other_test':
            # 实验室其他同学的测试集
            file_name = fr'src/torchData/otherTestData/other_test_torch_mat-{self.axis}.npy'
        else:
            file_name = fr'src/torchData/trainingData/{self.data_category}/{self.data_category}_torch_mat-{self.axis}.npy'


        dataSet = np.load(file_name)
        standScaler = StandardScaler(with_mean=True, with_std=True)
        print(f'正则，特征提取，降维-fileName = {file_name}, dataSet shape =({dataSet.shape})')

        columns = ['c' + str(i) for i in range(0, 36)]
        columns_stand = ['c' + str(i) for i in range(0, (7 * int(self.axis[0])))]

        df_stand_data = DataFrame(columns=columns_stand)
        df_features_data = []

        with tqdm(total=len(dataSet)) as pbar:  # 设置进度条
            for data in dataSet:
                pbar.update(1)  # 更新进度条
                label = data[-1]
                data = data[:-1].reshape(-1, 7 * int(self.axis[0])).T  # 转置，从[36,7*axis]转为[7*axis,36]

                data = standScaler.fit_transform(data)
                df_stand = DataFrame(data, columns=columns)

                df_features = self.extractFeatures(df_stand, columns)
                df_features = np.append(df_features, label)  # len = 36列*5+1=181

                df_stand_data = df_stand_data.append(DataFrame(np.array(df_stand).T,columns=columns_stand), ignore_index=True)
                df_features_data.append(df_features)

        df_features_data = np.array(df_features_data)
        # print(df_features_data.shape)
        dfStandSavePath = fr'src/ml_standData/{self.data_category}_stand_mat-{self.axis}.csv'
        featuresSavePath = fr'src/ml_tf_Features/{self.data_category}_features_mat-{self.axis}.npy'

        # 数据降维
        pca_feature = self.decomposition(df_features_data, self.axis, self.data_category)  # 列宽52 + 1label:

        df_stand_data.to_csv(dfStandSavePath)
        np.save(featuresSavePath, pca_feature)

    def extractFeatures(self, data: DataFrame, columns):
        # 均值 A,协方差C，峰值K,偏度S，
        dataA = np.array(data.apply(np.average, axis=0))
        dataC = np.array([x for x in data.apply(np.cov, axis=0).values])
        # 分别使用df.kurt()方法和df.skew()即可完成峰度he偏度计算
        dataK = np.array(data.kurt(axis=0))
        dataS = np.array(data.skew(axis=0))

        dataF = np.array(self.fft_T_function(data, columns))

        df_features = np.concatenate((dataA, dataC, dataK, dataS, dataF))  # len = 36列*5=180
        return df_features

    # 快速傅里叶变换
    def fft_T_function(self, dataMat, columns):
        '''
        axis = 0 垂直 做变换
        :param dataMat:
        :return:
        '''
        # dataMat = dataMat.T
        dataF = dataMat.apply(np.fft.fft, axis=0)
        data = dataF.apply(lambda x: np.abs(np.array(x).real), axis=0)
        return data.max(axis=0)

        # 数据降维

    def decomposition(self, X, axis, data_category, n_components=0.90):
        '''
            对实验数据降维
        :param X:数据集
        :return:X_pca
        '''
        if data_category == 'train':
            de_model = PCA(n_components=n_components, svd_solver='auto', whiten=True)

            X_data = X[:, :-1]
            X_label = X[:, -1]

            de_model.fit(X_data)
            joblib.dump(de_model, f'src/ml_pca_tf_model/pca_model{axis}.pkl')
            X_de = de_model.transform(X_data)
            X_de = DataFrame(X_de, columns=['pca' + str(i) for i in np.arange(len(X_de[0]))]).round(6)
            X_de['SPE'] = X_label

            # 它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分,
            # 返回各个成分各自的方差百分比(贡献率) = 0.95
            pca_feature = np.array(X_de)

            print(f"当前降维参数:{data_category}{axis},维度是{len(pca_feature[0]) - 1},data shape:{pca_feature.shape}")
            print('成分各自的方差百分比(贡献率):{}'.format(np.add.reduce(de_model.explained_variance_ratio_)))

            print(np.array(de_model.explained_variance_ratio_))


        else:
            pca_model = joblib.load(f'src/ml_pca_tf_model/pca_model{self.axis}.pkl')

            test_data = X[:, :-1]
            test_label = X[:, -1]
            test_pca_feature = pca_model.transform(test_data)
            test_pca_feature = DataFrame(test_pca_feature,
                                         columns=['pca' + str(i) for i in np.arange(len(test_pca_feature[0]))]).round(6)
            test_pca_feature['SPE'] = test_label
            pca_feature = np.array(test_pca_feature)

            print(f"当前降维参数:{data_category}{axis},维度是{len(pca_feature[0]) - 1},data shape:{pca_feature.shape}")

        return pca_feature


if __name__ == '__main__':
    axiss = ['6axis', '9axis']  # 36,42  36,63
    for axis in axiss:
        if platform.system() == 'Windows':
            action_root_path = f'D:/home/DataRec/action_windows-{axis}'
        else:
            action_root_path = f'/home/yanjilong/dataSets/DataRec/action_windows-{axis}'

        dataToTorch = DataToTorch(action_root_path, axis)
        # dataToTorch.readWindowsToTorchData()

    # # 读取数据显示
    # train_action_data_sets = ActionDataSets(data_category='train', axis='9axis')
    # for data, label in train_action_data_sets:
    #     print(data.shape)
    #     print(label)
    #     break
    # train_action_data_loader = DataLoader(train_action_data_sets, batch_size=64, shuffle=True, num_workers=2)
    #
    # for batch_data in train_action_data_loader:
    #     inputs, labels = batch_data
    #     label = torch.LongTensor([int(labels[0])])
    #     # print(inputs)
    #     print(labels.data)
    #     break

    # 从numpy数据中读取数据，经过正则化之后，提取时频特征,用于机器学习算法分类，不需要验证集
    for data_category in ['train', 'test']:
        for axis in ['9axis', '6axis']:
            standAndExtractFeatures = StandAndExtractTfFeatures(data_category, axis)
            standAndExtractFeatures.saveStandScalerData()

    # 实验室其它同学的测试集数据
    for axis in ['9axis', '6axis']:
        torch_data_path = f'src/torchData/otherTestData/other_test_torch_mat-{axis}.npy'
        standAndExtractFeatures = StandAndExtractTfFeatures('other_test', axis)
        standAndExtractFeatures.saveStandScalerData()
