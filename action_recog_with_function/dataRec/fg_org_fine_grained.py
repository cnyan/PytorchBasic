# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/31 17:48
@Describe：
    获取细粒度特征
    通过kmeans获取簇心
    使用余弦相似度判断动作相似度
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn
from dataToTorch import ActionDataSets
import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')



class Extract_origin_features():
    def __init__(self, axis, data_category):
        super(Extract_origin_features, self).__init__()

        self.axis = axis
        self.data_category = data_category
        action_data_set = ActionDataSets(data_category, axis)

        # 按批加载 pyTorch张量
        self.action_train_data_gen = DataLoader(action_data_set, shuffle=True,
                                                num_workers=2)  # 分成数组（len/128）个batch，每个batch长度是128
        print(f'{data_category}data shape: ({len(action_data_set)}{(action_data_set.data_shape())})')


    def extract_features(self):
        print(f'==============  提取{self.data_category}-{self.axis}的卷积特征=============')
        features_action0 = []
        features_action1 = []
        features_action2 = []
        features_action3 = []
        features_action4 = []

        for inputs in self.action_train_data_gen:

            data, label = inputs
            label = int(label)
            data = data.detach().cpu().numpy()

            fusion_features = data.flatten().tolist()

            if label == 0:
                features_action0.append(fusion_features)
            elif label == 1:
                features_action1.append(fusion_features)
            elif label == 2:
                features_action2.append(fusion_features)
            elif label == 3:
                features_action3.append(fusion_features)
            elif label == 4:
                features_action4.append(fusion_features)
            else:
                print(label)

        features_action0 = np.array(features_action0)
        features_action1 = np.array(features_action1)
        features_action2 = np.array(features_action2)
        features_action3 = np.array(features_action3)
        features_action4 = np.array(features_action4)

        print(f'features_action0.shape:{features_action0.shape}')
        print(f'features_action1.shape:{features_action1.shape}')
        print(f'features_action2.shape:{features_action2.shape}')
        print(f'features_action3.shape:{features_action3.shape}')
        print(f'features_action4.shape:{features_action4.shape}')

        np.save(f'src/fine_org_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action0.npy',
                features_action0)
        np.save(f'src/fine_org_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action1.npy',
                features_action1)
        np.save(f'src/fine_org_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action2.npy',
                features_action2)
        np.save(f'src/fine_org_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action3.npy',
                features_action3)
        np.save(f'src/fine_org_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action4.npy',
                features_action4)


class Kmeans_fine_grained():
    def __init__(self, axis, action_name, data_category):

        self.axis = axis
        self.action_name = action_name
        self.data_category = data_category

        if data_category == 'train':
            fusion_features_path = f'src/fine_org_grained_features/conv1d_2d_features/{data_category}_features_{axis}_{action_name}.npy'
            self.fusion_features = np.load(fusion_features_path)
        else:
            fusion_features_path = f'src/fine_org_grained_features/conv1d_2d_features/test_features_{axis}_{action_name}.npy'
            self.fusion_features = np.load(fusion_features_path)

        self.Tsne = TSNE(n_components=3, init='pca', random_state=0)
        self.Kmeans = KMeans(n_clusters=7, random_state=0)

    def get_tsne_data(self):
        """
        获取降维后的节点数据
        :return:
        """
        print(f'======== 获取tsne降维数据 {self.data_category}_{self.axis}_{self.action_name} =============')
        data = []
        targets = []
        for fusion_feature in self.fusion_features:  # 动作数据
            fusion_feature = fusion_feature.reshape(-1, int(self.axis[0]), 36)  # (7, 6, 36)
            for label, feature in enumerate(fusion_feature):
                feature = feature.flatten()
                targets.append(label)
                data.append(feature)

        data = np.array(data)
        targets = np.array(targets)
        print(data.shape)

        tsne_data = self.Tsne.fit_transform(data)
        # self.matplotlib(tsne_data)

        tsne_data_targets = []
        for index, tsne in enumerate(tsne_data):
            data_target = np.append(tsne, targets[index])
            tsne_data_targets.append(data_target.tolist())

        tsne_data_targets = np.array(tsne_data_targets)

        print(tsne_data_targets.shape)
        print(f'{self.data_category}_tsne_data_{self.axis}_{self.action_name} shape:{tsne_data_targets.shape}')
        np.save(
            f'src/fine_org_grained_features/tsne_data/{self.data_category}_tsne_data_{self.axis}_{self.action_name}.npy',
            tsne_data_targets)

    def train_kmeans(self):
        data_targets_path = f'src/fine_org_grained_features/tsne_data/train_tsne_data_{self.axis}_{self.action_name}.npy'
        data_targets = np.load(data_targets_path)
        tsne_data = data_targets[:, :3]
        tsne_targets = data_targets[:, 3]

        kmeans_model = self.Kmeans.fit(tsne_data)
        joblib.dump(kmeans_model,
                    f'src/fine_org_grained_features/kmeans_model/kmeans_model_{self.axis}_{self.action_name}.pkl')

        kmeans_cluster = kmeans_model.cluster_centers_  # 聚类的核心
        predicted = kmeans_model.predict(tsne_data)
        kmeans_cluster_label_dict = {}  # 保存聚类后的簇心，和标签值

        # 排列标签
        labels = np.zeros_like(predicted)
        for i in range(7):
            mask = (predicted == i)
            labels[mask] = mode(tsne_targets[mask])[0]
            kmeans_cluster_label = int(mode(tsne_targets[mask])[0][0])
            kmeans_cluster_label_dict[f'sensor-{kmeans_cluster_label}'] = kmeans_cluster[i]
        # print(kmeans_cluster_label_dict)

        np.save(
            f'src/fine_org_grained_features/cluster_label_dict/cluster_label_dict_{self.axis}_{self.action_name}.npy',
            kmeans_cluster_label_dict)

        # 计算准确度
        accuracy = accuracy_score(tsne_targets, labels)
        print(f'train_{self.axis}_{self.action_name} accuracy:{accuracy}')

    def predict_kmeans(self):
        data_targets_path = f'src/fine_org_grained_features/tsne_data/test_tsne_data_{self.axis}_{self.action_name}.npy'
        data_targets = np.load(data_targets_path)
        tsne_data = data_targets[:, :3]
        tsne_targets = data_targets[:, 3]

        kmeans_model = joblib.load(
            f'src/fine_org_grained_features/kmeans_model/kmeans_model_{self.axis}_{self.action_name}.pkl')

        predicted = kmeans_model.predict(tsne_data)

        # 排列标签
        labels = np.zeros_like(predicted)
        for i in range(7):
            mask = (predicted == i)
            labels[mask] = mode(tsne_targets[mask])[0]

        # 计算准确度
        accuracy = accuracy_score(tsne_targets, labels)
        print(f'test_{self.axis}_{self.action_name} accuracy:{accuracy}')


class Matplotlib_tsne():
    def __init__(self):
        pass

    def matplotlib(self):
        print(f'======== 绘制三维图像  =============')
        axiss = ['6axis', '9axis']
        actions_all = ['action0', 'action1', 'action2', 'action3', 'action4']

        for axis in axiss:
            plt.figure(figsize=(20, 14), dpi=108)
            plt.style.use('seaborn')
            for index, action_name in enumerate(actions_all):
                data_targets_path = f'src/fine_org_grained_features/tsne_data/train_tsne_data_{axis}_{action_name}.npy'
                data_targets = np.load(data_targets_path)

                tsne_data_node0 = np.array([x for x in data_targets if x[3] == 0])
                tsne_data_node1 = np.array([x for x in data_targets if x[3] == 1])
                tsne_data_node2 = np.array([x for x in data_targets if x[3] == 2])
                tsne_data_node3 = np.array([x for x in data_targets if x[3] == 3])
                tsne_data_node4 = np.array([x for x in data_targets if x[3] == 4])
                tsne_data_node5 = np.array([x for x in data_targets if x[3] == 5])
                tsne_data_node6 = np.array([x for x in data_targets if x[3] == 6])

                # ax = Axes3D(fig)
                # plt.title=('{self.action_name}-{self.axis} scatter plot')
                ax = plt.subplot(2, 3, index + 1, projection='3d')
                # 调整视角
                ax.view_init(elev=10, azim=20)  # 仰角,方位角

                ax.scatter(tsne_data_node0[:, 0], tsne_data_node0[:, 1], tsne_data_node0[:, 2], c='r', label='sensor-0')
                ax.scatter(tsne_data_node1[:, 0], tsne_data_node1[:, 1], tsne_data_node1[:, 2], c='y', label='sensor-1')
                ax.scatter(tsne_data_node2[:, 0], tsne_data_node2[:, 1], tsne_data_node2[:, 2], c='m', label='sensor-2')
                ax.scatter(tsne_data_node3[:, 0], tsne_data_node3[:, 1], tsne_data_node3[:, 2], c='g', label='sensor-3')
                ax.scatter(tsne_data_node4[:, 0], tsne_data_node4[:, 1], tsne_data_node4[:, 2], c='b', label='sensor-4')
                ax.scatter(tsne_data_node5[:, 0], tsne_data_node5[:, 1], tsne_data_node5[:, 2], c='k', label='sensor-5')
                ax.scatter(tsne_data_node6[:, 0], tsne_data_node6[:, 1], tsne_data_node6[:, 2], c='orange',
                           label='sensor-6')

                # ax.set_title(f'{self.action_name}-{self.axis} scatter plot')
                ax.set_zlabel('X', fontsize=20)  # 坐标轴
                ax.set_ylabel('Y', fontsize=20)
                ax.set_xlabel('Z', fontsize=20)
                ax.legend()
                plt.subplots_adjust(hspace=0.4)
                plt.title(f'{action_name}', fontsize=30)
                plt.legend()
            plt.savefig(f'src/fine_org_grained_features/tsne_plt/tense_plt_scatter-{axis}.jpg')
            plt.show()
            plt.close()


if __name__ == '__main__':

    axis_all = ['6axis', '9axis']
    data_categorys = ['train', 'test']
    for axis in axis_all:
        for data_category in data_categorys:
            # 抽取训练集、测试集多维度卷积融合特征
            extractFeatures = Extract_origin_features(axis, data_category=data_category)
            # extractFeatures.extract_features()

    actions_all = ['action0', 'action1', 'action2', 'action3', 'action4']

    for axis in axis_all:
        for action_name in actions_all:
            # 降维后做kmeans训练
            kmeans_fine_grained = Kmeans_fine_grained(axis, action_name, data_category='train')
            # kmeans_fine_grained.get_tsne_data()
            kmeans_fine_grained.train_kmeans()

    for axis in axis_all:
        for action_name in actions_all:
            # 降维后做kmeans训练
            kmeans_fine_grained = Kmeans_fine_grained(axis, action_name, data_category='test')
            # kmeans_fine_grained.get_tsne_data()
            kmeans_fine_grained.predict_kmeans()

    # 绘制三维视图
    matplotlib_tsne = Matplotlib_tsne()
    # matplotlib_tsne.matplotlib()
