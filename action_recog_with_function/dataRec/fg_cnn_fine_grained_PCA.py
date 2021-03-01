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
import torch.nn.functional as F
from sklearn.decomposition import PCA
import math
import joblib
from scipy.stats import mode
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from dataToTorch import ActionDataSets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings('ignore')

standScaler = StandardScaler(with_mean=True, with_std=True)


class MyMultiTempSpaceConfluenceNet(nn.Module):
    """
    时空卷积融合
    """

    def __init__(self, axis):
        super(MyMultiTempSpaceConfluenceNet, self).__init__()

        self.axis = axis
        self.temporal1_layer = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 2, dilation=2),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv1d(256, 36, 1, 1, 0),
            nn.BatchNorm1d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 输入大小（7*axis，36）
        self.spatial2_layer = nn.Sequential(
            nn.Conv2d(1, 128, 1, 1, 0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1, 1, 0),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            # nn.AvgPool1d(2, 2)
        )  # 128*18

        self.confluence3_layer = nn.Sequential(
            nn.Conv2d(1, 128, 2, 1, 1),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )  # 256*9

        self.confluence4_layer = nn.Sequential(
            nn.Conv2d(128, 256, 2, 1, 1),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )  # 256*9

        self.classifier = nn.Sequential(
            nn.Linear(256 * (math.ceil((7 * axis) / 4)) * 9, 5),
        )

    def forward(self, x):
        x_1d = x.permute([0, 2, 1])
        temp = self.temporal1_layer(x_1d)
        temp = temp.unsqueeze(0).permute([1, 0, 3, 2])

        x_2d = x.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度
        space = self.spatial2_layer(x_2d)

        input = temp + space  # [64, 1, 42, 36]
        # input_2d = input.unsqueeze(0).permute([1, 0, 2, 3])  # 扩展一个维度

        out = self.confluence3_layer(input)
        out = self.confluence4_layer(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class Extract_1D_2D_features():
    def __init__(self, modelNet, model_name, axis, data_category):
        super(Extract_1D_2D_features, self).__init__()

        self.model = modelNet
        self.conv1d_features = {}
        self.conv2d_features = {}

        self.model.temporal1_layer.register_forward_hook(self.get_conv1d_activation("temporal1_layer"))
        self.model.spatial2_layer.register_forward_hook(self.get_conv1d_activation("spatial2_layer"))

        self.axis = axis
        self.data_category = data_category
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{axis}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.model.eval()
        action_data_set = ActionDataSets(data_category, axis)

        # 按批加载 pyTorch张量
        self.action_train_data_gen = DataLoader(action_data_set, shuffle=True,
                                                num_workers=2)  # 分成数组（len/128）个batch，每个batch长度是128
        print(f'{data_category}data shape: ({len(action_data_set)}{(action_data_set.data_shape())})')

    def get_conv1d_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv1d_features[name] = output.detach().cpu()

        return hook

    def get_conv2d_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv2d_features[name] = output.detach().cpu()

        return hook

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

            output = self.model(data)

            conv1d_features = self.conv1d_features["temporal1_layer"]
            conv2d_features = self.conv1d_features["spatial2_layer"]
            conv1d_features = conv1d_features.permute([0, 2, 1]).squeeze(0).numpy()
            conv2d_features = conv2d_features.permute([1, 0, 2, 3]).squeeze(0).squeeze(0).numpy()

            # print(conv1d_features.shape)
            # print(conv2d_features.shape)

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

        np.save(f'src/fine_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action0.npy',
                features_action0)
        np.save(f'src/fine_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action1.npy',
                features_action1)
        np.save(f'src/fine_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action2.npy',
                features_action2)
        np.save(f'src/fine_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action3.npy',
                features_action3)
        np.save(f'src/fine_grained_features/conv1d_2d_features/{self.data_category}_features_{self.axis}_action4.npy',
                features_action4)


class Kmeans_fine_grained():
    def __init__(self, axis, action_name, data_category):

        self.axis = axis
        self.action_name = action_name
        self.data_category = data_category

        if data_category == 'train':
            fusion_features_path = f'src/fine_grained_features/conv1d_2d_features/{data_category}_features_{axis}_{action_name}.npy'
            self.fusion_features = np.load(fusion_features_path)
        else:
            fusion_features_path = f'src/fine_grained_features/conv1d_2d_features/test_features_{axis}_{action_name}.npy'
            self.fusion_features = np.load(fusion_features_path)

        self.pca = PCA(n_components=3, svd_solver='auto', whiten=True)

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
            fusion_feature = fusion_feature.reshape(-1, int(self.axis[0]), 36)  # (7, 6, 36) 给一个窗口数据，按节点编号打标签
            for label, feature in enumerate(fusion_feature):
                feature = feature.flatten()
                targets.append(label)
                data.append(feature)

        data = np.array(data)
        targets = np.array(targets)
        print(data.shape)

        if self.data_category == 'train':
            pca_fit = self.pca.fit(data)
            # tsne_params = tsne_fit.get_params()
            joblib.dump(pca_fit,f'src/fine_grained_features/tsne_model/pca_model_{self.axis}.pkl')
        else:
            self.pca = joblib.load(f'src/fine_grained_features/tsne_model/pca_model_{self.axis}.pkl')

        tsne_data = self.pca.transform(data)

        # self.matplotlib(tsne_data)

        tsne_data_targets = []
        for index, tsne in enumerate(tsne_data):
            data_target = np.append(tsne, targets[index])
            tsne_data_targets.append(data_target.tolist())

        tsne_data_targets = np.array(tsne_data_targets)

        print(tsne_data_targets.shape)
        print(f'{self.data_category}_tsne_pca_data_{self.axis}_{self.action_name} shape:{tsne_data_targets.shape}')
        np.save(
            f'src/fine_grained_features/tsne_data/{self.data_category}_pca_data_{self.axis}_{self.action_name}.npy',
            tsne_data_targets)

    def train_kmeans(self):
        data_targets_path = f'src/fine_grained_features/tsne_data/train_pca_data_{self.axis}_{self.action_name}.npy'
        data_targets = np.load(data_targets_path)
        tsne_data = data_targets[:, :3]
        tsne_targets = data_targets[:, 3]

        kmeans_model = self.Kmeans.fit(tsne_data)
        joblib.dump(kmeans_model,
                    f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_{self.action_name}.pkl')

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
            f'src/fine_grained_features/cluster_label_dict/cluster_label_pca_dict_{self.axis}_{self.action_name}.npy',
            kmeans_cluster_label_dict)

        # 计算准确度
        accuracy = accuracy_score(tsne_targets, labels)
        print(f'train_{self.axis}_{self.action_name} accuracy:{accuracy}')

    def predict_kmeans(self):
        data_targets_path = f'src/fine_grained_features/tsne_data/test_pca_data_{self.axis}_{self.action_name}.npy'
        data_targets = np.load(data_targets_path)
        tsne_data = data_targets[:, :3]
        tsne_targets = data_targets[:, 3]

        kmeans_model = joblib.load(
            f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_{self.action_name}.pkl')

        predicted = kmeans_model.predict(tsne_data)

        # 排列标签
        labels = np.zeros_like(predicted)
        for i in range(7):
            mask = (predicted == i)
            labels[mask] = mode(tsne_targets[mask])[0]

        # 计算准确度
        accuracy = accuracy_score(tsne_targets, labels)
        print(f'test_{self.axis}_{self.action_name} accuracy:{accuracy}')


class FG__vector_Predict_with_kmeans():
    def __init__(self, axis, action_name):
        self.axis = axis
        # self.action_name = action_name

        self.pca = joblib.load(f'src/fine_grained_features/tsne_model/pca_model_{self.axis}.pkl')

        self.kmeans_model_action0 = joblib.load(
            f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_action0.pkl')
        self.kmeans_model_action1 = joblib.load(
            f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_action1.pkl')
        self.kmeans_model_action2 = joblib.load(
            f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_action2.pkl')
        self.kmeans_model_action3 = joblib.load(
            f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_action3.pkl')
        self.kmeans_model_action4 = joblib.load(
            f'src/fine_grained_features/kmeans_model/kmeans_model_pca_{self.axis}_action4.pkl')

        self.modelNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))
        self.modelNet.load_state_dict(torch.load(f'src/model/myMultiTempSpaceConfluenceNet_{axis}_model.pkl', map_location='cpu'))
        self.modelNet.eval()

        self.conv1d_features = {}
        self.conv2d_features = {}
        self.modelNet.temporal1_layer.register_forward_hook(self.get_conv1d_activation("temporal1_layer"))
        self.modelNet.spatial2_layer.register_forward_hook(self.get_conv1d_activation("spatial2_layer"))

        self.toTensor = transforms.ToTensor()

        self.all_cluster_label_dict = np.load(
            f'src/fine_grained_features/cluster_label_dict/all_cluster_label_pca_dict_{self.axis}.npy',
            allow_pickle=True).item()
        # print(self.all_cluster_label_dict)

    def calculate_fg_with_test_window_data(self):
        data_targets_path = f'D:/home/DataRec/action_windows-{self.axis}/action_window_1.csv'

        data_mat = pd.read_csv(data_targets_path, dtype=float, header=0).round(3)
        data_mat = np.array(data_mat)[:, 1:-1]

        data_mat = data_mat[:int(len(data_mat) / 36) * 36, :]  # 确保是窗口长度的倍数
        data_mat = np.reshape(data_mat, (-1, 36, int(self.axis[0]) * 7))
        # print(data_mat.shape)
        data_mat = data_mat[:5, :, ]  # 每个动作取200个

        for df in data_mat:
            data = np.array(df)[:, :].T  # 转为42*36  data = np.array(df)[2:-2, :].T  # 转为36*63
            data = standScaler.fit_transform(data)
            data = self.toTensor(data)
            data = data.to(torch.float32)
            # data = data.unsqueeze(0)  # 扩展一个维度
            output = self.modelNet(data)

            prob = F.softmax(output, dim=1)
            action_score = torch.max(prob, 1)[0].data.numpy()[0]
            action_class = torch.max(prob, 1)[1].data.numpy()[0]  # 最大值下标

            # 计算细粒度
            conv1d_features = self.conv1d_features["temporal1_layer"]
            conv2d_features = self.conv1d_features["spatial2_layer"]
            conv1d_features = conv1d_features.permute([0, 2, 1]).squeeze(0).numpy()
            conv2d_features = conv2d_features.permute([1, 0, 2, 3]).squeeze(0).squeeze(0).numpy()

            fusion_features = conv1d_features + conv2d_features
            fusion_features = fusion_features.reshape(-1, int(self.axis[0]), 36)

            true_node_labels = []
            fusion_data = []
            for node_label, feature in enumerate(fusion_features):
                feature = feature.flatten()
                true_node_labels.append(node_label)
                fusion_data.append(feature)

            tsne_data = self.pca.transform(np.array(fusion_data))
            tsne_data = np.array(tsne_data, dtype=np.float16)
            # print(tsne_data)

            cluster_label_dict = self.all_cluster_label_dict[f'action{action_class}']
            print(cluster_label_dict)

            vector_scores = []
            for index in range(7):
                sensor_num = f'sensor-{index}'
                # 余弦相似度
                vector_score = self.cosine_similarity_method(tsne_data[index], cluster_label_dict[sensor_num])
                # vector_score = self.distance_seuclidean(tsne_data[index], cluster_label_dict[sensor_num])
                vector_scores.append(round(vector_score, 1))

            print(vector_scores)

            # if action_class == 0:
            #     kmeans_predicted = self.kmeans_model_action0.predict(tsne_data)
            # elif action_class == 1:
            #     kmeans_predicted = self.kmeans_model_action1.predict(tsne_data)
            # elif action_class == 2:
            #     kmeans_predicted = self.kmeans_model_action2.predict(tsne_data)
            # elif action_class == 3:
            #     kmeans_predicted = self.kmeans_model_action3.predict(tsne_data)
            # elif action_class == 4:
            #     kmeans_predicted = self.kmeans_model_action4.predict(tsne_data)
            # else:
            #     kmeans_predicted = 'error'
            #
            # vector_scores = []
            # for index,kmeans_v in enumerate(kmeans_predicted):
            #     sensor_num = f'sensor-{kmeans_v}'
            #     # 余弦相似度
            #     vector_score = self.cosine_similarity(tsne_data[index], cluster_label_dict[sensor_num])
            #     # vector_score = self.distance_seuclidean(tsne_data[index], cluster_label_dict[sensor_num])
            #     vector_scores.append(round(vector_score, 1))
            #
            # print(vector_scores)

    def distance_seuclidean(self, x, y):
        X = np.vstack([x, y])
        distance = pairwise_distances(X, metric='seuclidean')
        # distance = pairwise_distances(X, metric='mahalanobis')
        return distance[0][1]

    def cosine_similarity_method(self, x, y, norm=True):
        """ 计算两个向量x和y的余弦相似度 """
        # 计算余弦相似度
        # assert len(x) == len(y), "len(x) != len(y)"
        # zero_list = [0] * len(x)
        # zero_list = np.array(zero_list)
        # if x.all() == zero_list.all() or y.all() == zero_list.all():
        #     return float(1) if x.all() == y.all() else float(0)
        #
        # # method 1
        # res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        # cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

        X = np.vstack([np.abs(x), np.abs(y)])
        cos = cosine_similarity(X)[0][1]
        return cos
        # return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    def get_conv1d_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv1d_features[name] = output.detach().cpu()

        return hook

    def get_conv2d_activation(self, name):
        # 定义钩子
        def hook(model, input, output):
            self.conv2d_features[name] = output.detach().cpu()

        return hook


class Get_cluster_label_dict():
    """
    将不同动作、不同传感器的kmeans簇新组合成一个完整的字典，用于在线识别
    """

    def __init__(self, axis):
        super(Get_cluster_label_dict, self).__init__()
        self.axis = axis

    def getClusterLabelDict(self):
        actions_all = ['action0', 'action1', 'action2', 'action3', 'action4']
        all_cluster_label_dict = {}
        for action_name in actions_all:
            dict_path = f'src/fine_grained_features/cluster_label_dict/cluster_label_pca_dict_{self.axis}_{action_name}.npy'
            cluster_label_dict = np.load(dict_path, allow_pickle=True).item()
            all_cluster_label_dict[action_name] = cluster_label_dict
        np.save(f'src/fine_grained_features/cluster_label_dict/all_cluster_label_pca_dict_{self.axis}.npy',
                all_cluster_label_dict)


class Matplotlib_tsne():
    def __init__(self):
        pass

    def matplotlib(self):
        print(f'======== 绘制三维图像  =============')
        axiss = ['6axis', '9axis']
        actions_all = ['action0', 'action1', 'action2', 'action3', 'action4']

        for axis in axiss:
            plt.figure(figsize=(20, 14), dpi=516)
            plt.style.use('seaborn')
            for index, action_name in enumerate(actions_all):
                data_targets_path = f'src/fine_grained_features/tsne_data/train_pca_data_{axis}_{action_name}.npy'
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
                ax.set_zlabel('Z', fontsize=20)  # 坐标轴
                ax.set_ylabel('Y', fontsize=20)
                ax.set_xlabel('X', fontsize=20)
                ax.legend()
                plt.subplots_adjust(hspace=0.4)
                plt.title(f'{action_name}', fontsize=30)
                plt.legend()
            plt.savefig(f'src/fine_grained_features/tsne_plt/tense_pca_plt_scatter-{axis}.jpg')
            plt.show()
            plt.close()


if __name__ == '__main__':

    axis_all = ['6axis', '9axis']
    data_categorys = ['train', 'test']
    for axis in axis_all:
        myMultiTempSpaceConfluenceNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))

        models_all = {'myMultiTempSpaceConfluenceNet': myMultiTempSpaceConfluenceNet}

        for model_name, model in models_all.items():
            for data_category in data_categorys:
                # 抽取训练集、测试集多维度卷积融合特征
                extractFeatures = Extract_1D_2D_features(model, model_name, axis, data_category=data_category)
                # extractFeatures.extract_features()

    actions_all = ['action0', 'action1', 'action2', 'action3', 'action4']

    for axis in axis_all:
        for action_name in actions_all:
            # 降维后做kmeans训练
            kmeans_fine_grained = Kmeans_fine_grained(axis, action_name, data_category='train')
            kmeans_fine_grained.get_tsne_data()
            kmeans_fine_grained.train_kmeans()

    for axis in axis_all:
        for action_name in actions_all:
            # 降维后做kmeans训练
            kmeans_fine_grained = Kmeans_fine_grained(axis, action_name, data_category='test')
            kmeans_fine_grained.get_tsne_data()
            kmeans_fine_grained.predict_kmeans()

    # 绘制三维视图
    matplotlib_tsne = Matplotlib_tsne()
    matplotlib_tsne.matplotlib()

    for axis in axis_all:
        get_cluster_label_dict = Get_cluster_label_dict(axis)
        get_cluster_label_dict.getClusterLabelDict()

    fg_vector_Predict_with_kmeans = FG__vector_Predict_with_kmeans('6axis', 'action0')
    fg_vector_Predict_with_kmeans.calculate_fg_with_test_window_data()
