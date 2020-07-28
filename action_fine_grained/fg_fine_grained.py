# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/28 9:19
@Describe：

"""
import math
from fg_data_input import Input_Data
import torch
import os
import numpy as np


class Grained():
    def __init__(self, model_name='1'):
        self.feature_node = []  # 节点特征向量

    def get_grained(self, model_name='1'):
        model = torch.load('src/model/fg_cnn_model_' + model_name + '.pkl')
        model.eval()
        test_action_data_set = Input_Data("src/nodeData/test/test_action_data_" + model_name + ".npy")

        feature_vector_action_0 = []  # 动作-1特征向量
        feature_vector_action_1 = []  # 动作-2特征向量
        feature_vector_action_2 = []  # 动作-3特征向量
        feature_vector_action_3 = []  # 动作-4特征向量
        feature_vector_action_4 = []  # 动作-5特征向量

        for data, label in test_action_data_set:
            data = data.unsqueeze(0)  # 扩展一个维度
            feature_output = model(data).detach().numpy()[0]
            if label == 0:
                feature_vector_action_0.append(feature_output)
            elif label == 1:
                feature_vector_action_1.append(feature_output)
            elif label == 2:
                feature_vector_action_2.append(feature_output)
            elif label == 3:
                feature_vector_action_3.append(feature_output)
            elif label == 4:
                feature_vector_action_4.append(feature_output)
        feature_vector_action_0 = np.array(feature_vector_action_0).mean(axis=0)
        feature_vector_action_1 = np.array(feature_vector_action_1).mean(axis=0)
        feature_vector_action_2 = np.array(feature_vector_action_2).mean(axis=0)
        feature_vector_action_3 = np.array(feature_vector_action_3).mean(axis=0)
        feature_vector_action_4 = np.array(feature_vector_action_4).mean(axis=0)

        self.feature_node.append(feature_vector_action_0)
        self.feature_node.append(feature_vector_action_1)
        self.feature_node.append(feature_vector_action_2)
        self.feature_node.append(feature_vector_action_3)
        self.feature_node.append(feature_vector_action_4)
        return self.feature_node


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    # fusion_arr = np.append(x, y).reshape(2, -1)
    # mean_fusion_arr = np.mean(fusion_arr, axis=0)
    # x = (x - mean_fusion_arr).tolist()
    # y = (y - mean_fusion_arr).tolist()

    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


if __name__ == '__main__':
    grained = Grained()
    # all_node_feature： shape = (7, 5, 5), 7表示7个节点，每个节点对应5个动作，每个动作向量长度为5
    all_node_feature = np.array([])

    for i in ['1', '10', '2', '3', '4', '6', '7']:
        node_feature = np.array(grained.get_grained(i))
        all_node_feature = np.append(all_node_feature, node_feature)

    all_node_feature = all_node_feature.flatten().reshape(-1, 5, 5)  # shape=(7,5,5),7个节点，每个节点对应5个动作，每个动作向量长度为5
    file_path = 'src/grained/all_node_feature_vector.npy'
    if os.path.exists(file_path):
        os.remove(file_path)
    np.save(file_path, all_node_feature)
    print(np.load(file_path))

    # a = [1, 2]
    # b = [4, 5]
    # print(cosine_similarity(a, b)) # 0.9778
    # #
    # a = [-2, -1]
    # b = [1, 2]
    # print(cosine_similarity(a, b)) # -0.79999
    #
    # print(cosine_similarity([0, 0], [0, 0]))  # 1.0
    # print(cosine_similarity([1, 1], [0, 0]))  # 0.0
    # print(cosine_similarity([1, 1], [-1, -1]))  # -1.0
    # print(cosine_similarity([1, 1], [2, 2]))  # 1.0
    # print(cosine_similarity([3, 3], [4, 4]))  # 1.0
    # print(cosine_similarity([1, 2, 2, 1, 1, 1, 0], [1, 2, 2, 1, 1, 2, 1]))  # 0.938194187433
