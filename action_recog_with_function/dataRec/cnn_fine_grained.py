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


class MyMultiTempSpaceConfluenceNet(nn.Module):
    """
    时空卷积融合
    """

    def __init__(self, axis):
        super(MyMultiTempSpaceConfluenceNet, self).__init__()
        self.axis = axis
        self.temporal1_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 7 * axis, 1, 1, 0),
            nn.BatchNorm1d(7 * axis, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(7 * axis, 7 * axis, 5, 1, 2),
            nn.Dropout(0.5),
            nn.BatchNorm1d(7 * axis, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(7 * axis, 7 * axis, 1, 1, 0),
            nn.BatchNorm1d(7 * axis, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18
        self.spatial2_layer = nn.Sequential(
            nn.Conv2d(1, 1, 1, 1, 0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, 1, 0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18

        self.confluence3_layer = nn.Sequential(
            nn.Conv1d(7 * axis, 128, 2, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        )  # 256*9

        self.confluence4_layer = nn.Sequential(
            nn.Conv1d(128, 256, 2, 1, 1),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )  # 256*9

        self.classifier = nn.Sequential(
            nn.Linear(256 * 9, 5)
        )

    def forward(self, x):
        temp = self.temporal1_layer(x)

        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        spital = self.spatial2_layer(x_2d)
        spital = spital.permute([1, 0, 2, 3]).squeeze(0)

        out = self.confluence3_layer(temp + spital)

        out = self.confluence4_layer(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Extract_1D_2D_features():
    def __init__(self, modelNet, model_name, axis):
        super(Extract_1D_2D_features, self).__init__()

        self.model = modelNet
        self.conv1d_features = {}
        self.conv2d_features = {}

        self.model.temporal1_layer.register_forward_hook(self.get_conv1d_activation("temporal1_layer"))
        self.model.spatial2_layer.register_forward_hook(self.get_conv1d_activation("spatial2_layer"))

        self.axis = axis
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{axis}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.model.eval()
        action_data_train_set = ActionDataSets('train', axis)

        # 按批加载 pyTorch张量
        self.action_train_data_gen = DataLoader(action_data_train_set, shuffle=True,
                                                num_workers=2)  # 分成数组（len/128）个batch，每个batch长度是128
        print(f'train_data shape: ({len(action_data_train_set)}{(action_data_train_set.data_shape())})')

    def get_conv1d_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv1d_features[name] = output.detach().cpu().numpy()

        return hook

    def get_conv2d_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv2d_features[name] = output.detach().cpu().numpy()

        return hook

    def extract_features(self):
        print(f'==============  提取{self.axis}的卷积特征=============')
        features_action0 = []
        features_action1 = []
        features_action2 = []
        features_action3 = []
        features_action4 = []
        count0 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for inputs in self.action_train_data_gen:

            data, label = inputs
            label = int(label)

            output = self.model(data)

            conv1d_features = self.conv1d_features["temporal1_layer"]
            conv2d_features = self.conv1d_features["spatial2_layer"]
            fusion_features = conv1d_features + conv2d_features
            fusion_features = fusion_features.flatten().tolist()

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

        np.save(f'src/fine_grained_features/conv1d_2d_features/features_{self.axis}_action0.npy', features_action0)
        np.save(f'src/fine_grained_features/conv1d_2d_features/features_{self.axis}_action1.npy', features_action1)
        np.save(f'src/fine_grained_features/conv1d_2d_features/features_{self.axis}_action2.npy', features_action2)
        np.save(f'src/fine_grained_features/conv1d_2d_features/features_{self.axis}_action3.npy', features_action3)
        np.save(f'src/fine_grained_features/conv1d_2d_features/features_{self.axis}_action4.npy', features_action4)


class Kmeans_fine_grained():
    def __init__(self, axis, action_name):
        fusion_features_path = f'src/fine_grained_features/conv1d_2d_features/features_{axis}_{action_name}.npy'
        self.fusion_features = np.load(fusion_features_path)
        self.axis = axis
        self.action_name = action_name
        self.Tsne = TSNE(n_components=3, init='pca', random_state=0)
        self.Kmeans = KMeans(n_clusters=7, random_state=0)

    def get_tsne_data(self):
        """
        获取降维后的节点数据
        :return:
        """
        print(f'======== 获取tsne降维数据 {self.axis}_{self.action_name} =============')
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
        print(f'tsne_data_{self.axis}_{self.action_name} shape:{tsne_data_targets.shape}')
        np.save(f'src/fine_grained_features/tsne_data/tsne_data_{self.axis}_{self.action_name}.npy', tsne_data_targets)

    def train_kmeans(self):
        data_targets_path = f'src/fine_grained_features/tsne_data/tsne_data_{self.axis}_{self.action_name}.npy'
        data_targets = np.load(data_targets_path)
        tsne_data = data_targets[:, :3]
        tsne_targets = data_targets[:, 3]

        kmeans_model = self.Kmeans.fit(tsne_data)
        joblib.dump(kmeans_model,
                    f'src/fine_grained_features/kmeans_model/kmeans_model_{self.axis}_{self.action_name}.pkl')

        predicted = kmeans_model.predict(tsne_data)

        # 排列标签
        labels = np.zeros_like(predicted)
        for i in range(7):
            mask = (predicted == i)
            labels[mask] = mode(tsne_targets[mask])[0]

        # 计算准确度
        accuracy = accuracy_score(tsne_targets, labels)
        print(f'{self.axis}_{self.action_name} accuracy:{accuracy}')

    def predict_kmeans(self):
        pass


    def matplotlib(self):
        print(f'======== 绘制三维图像 {self.axis}_{self.action_name} =============')
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import seaborn

        data_targets_path = f'src/fine_grained_features/tsne_data/tsne_data_{self.axis}_{self.action_name}.npy'
        data_targets = np.load(data_targets_path)

        tsne_data_node0 = np.array([x for x in data_targets if x[3] == 0])
        tsne_data_node1 = np.array([x for x in data_targets if x[3] == 1])
        tsne_data_node2 = np.array([x for x in data_targets if x[3] == 2])
        tsne_data_node3 = np.array([x for x in data_targets if x[3] == 3])
        tsne_data_node4 = np.array([x for x in data_targets if x[3] == 4])
        tsne_data_node5 = np.array([x for x in data_targets if x[3] == 5])
        tsne_data_node6 = np.array([x for x in data_targets if x[3] == 6])


        fig = plt.figure()
        #ax = Axes3D(fig)
        # plt.title=('{self.action_name}-{self.axis} scatter plot')
        ax = plt.subplot(111, projection='3d')
        # 调整视角
        ax.view_init(elev=10, azim=20)  # 仰角,方位角

        ax.scatter(tsne_data_node0[:, 0], tsne_data_node0[:, 1], tsne_data_node0[:, 2], c='r', label='sensor-0')
        ax.scatter(tsne_data_node1[:, 0], tsne_data_node1[:, 1], tsne_data_node1[:, 2], c='y', label='sensor-1')
        ax.scatter(tsne_data_node2[:, 0], tsne_data_node2[:, 1], tsne_data_node2[:, 2], c='m', label='sensor-2')
        ax.scatter(tsne_data_node3[:, 0], tsne_data_node3[:, 1], tsne_data_node3[:, 2], c='g', label='sensor-3')
        ax.scatter(tsne_data_node4[:, 0], tsne_data_node4[:, 1], tsne_data_node4[:, 2], c='b', label='sensor-4')
        ax.scatter(tsne_data_node5[:, 0], tsne_data_node5[:, 1], tsne_data_node5[:, 2], c='k', label='sensor-5')
        ax.scatter(tsne_data_node6[:, 0], tsne_data_node6[:, 1], tsne_data_node6[:, 2], c='orange', label='sensor-6')

        # ax.set_title(f'{self.action_name}-{self.axis} scatter plot')
        ax.set_zlabel('X')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('Z')
        ax.legend()

        plt.title(f'{self.action_name}-{self.axis} scatter plot')
        plt.savefig(f'src/fine_grained_features/tsne_plt/tense_plt_{self.axis}_{self.action_name}.jpg')
        plt.legend()
        plt.show()
        plt.close()


if __name__ == '__main__':

    axis_all = ['6axis', '9axis']

    for axis in axis_all:
        myMultiTempSpaceConfluenceNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))

        models_all = {'myMultiTempSpaceConfluenceNet': myMultiTempSpaceConfluenceNet}

        for model_name, model in models_all.items():
            extractFeatures = Extract_1D_2D_features(model, model_name, axis)
            # extractFeatures.extract_features()

    actions_all = ['action0', 'action1', 'action2', 'action3', 'action4']

    for axis in axis_all:
        for action_name in actions_all:
            kmeans_fine_grained = Kmeans_fine_grained(axis, action_name)
            # kmeans_fine_grained.get_tsne_data()
            # kmeans_fine_grained.train_kmeans()
            # kmeans_fine_grained.matplotlib()
