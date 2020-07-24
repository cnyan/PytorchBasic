# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/10/16 16:23
@Describe：

"""

import numpy as np
import platform
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from torchvision import models

import math
import os
import glob
import time
# 通过混淆矩阵，分析错误的分类
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from cnn_action_nn import ActionsDataSet
from cnn_base_var import action_window_row, action_window_col

import warnings

warnings.filterwarnings('ignore')


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


# 自定义计算准确度的函数
def rightness(predictions, labels):
    prob = F.softmax(predictions, dim=1)
    score = torch.max(prob, 1)[0].cpu().data

    pred = torch.max(predictions, 1)[1]  # 最大值下标

    # print(f'--------{labels.data.numpy()[0]}')
    # print(f'--------{pred.data.numpy()[0]}')
    # if labels.data.numpy()[0] == pred.data.numpy()[0]:
    #     print('True')
    # else:
    #     print('False')

    rights = pred.eq(labels.data.view_as(pred)).sum()

    # if score < 0.6:
    #     isError = True  # 识别错误
    #     if pred.cpu().data.numpy()[0] == labels.data.numpy()[0]:
    #         isError = False  # 正确
    #     print('pred:{},label:{},score:{:.2f},isError:{}'.format(pred.cpu().data.numpy(), labels.data.numpy(), score,
    #                                                             isError))

    return rights.cpu(), len(labels), score, pred.data.numpy()


def imshow(inp):
    """show image for tensor
        对张量再次变形，并且将值 反  归一化
    """
    inp = inp.transpose((1, 2, 0))  # transpose 按照指定维度旋转矩阵
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # 反序列化
    inp = np.clip(inp, -1, 1)
    plt.imshow(inp)
    plt.show()


# 混淆矩阵
def confusionMatrix(y_test, y_predict, is_migrate):
    if is_migrate:
        file_name = 'plt_img/cm_resnet18_predict.jpg'
    else:
        file_name = 'plt_img/cm_cnn_predict.jpg'

    sns.set()
    mat = confusion_matrix(y_test, y_predict)
    sns.heatmap(mat, square=True, annot=True, cbar=True)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.savefig(file_name)
    plt.show()


class Predict:
    def __init__(self, file_path, is_migrate=False):
        super().__init__()
        self.file_path = file_path
        self.is_migrate = is_migrate

        self.size = (action_window_col, action_window_row)
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # 是否使用迁移网络
        if is_migrate:
            self.mode_path = 'src/cnn_model_migrate.pkl'
        else:
            self.mode_path = 'src/cnn_model.pkl'

        if is_migrate:  # 加载迁移网络，总是在GPU训练
            if torch.cuda.is_available():
                self.new_model = torch.load(self.mode_path)
            else:
                # 加载模型(GPU转CPU)
                # model_weights = torch.load(self.mode_path, map_location=lambda storage, loc: storage)
                model_weights = torch.load(self.mode_path)  # 加载CPU模型
                # model_weights = torch.load(self.mode_path, map_location='cpu')  # pytorch0.4.0及以上版本,与上等价
                # 构建网络架构(迁移网络)
                self.new_model = models.resnet18(pretrained=False)
                num_ftrs = self.new_model.fc.in_features  # 获取模型最后一层的输出
                self.new_model.fc = nn.Linear(num_ftrs, 5)  # 更改模型最后一层，输出类别为5
                self.new_model.load_state_dict(model_weights.state_dict())  # # 用weights初始化网络
        else:
            # 使用自定义网络（CPU与GPU独自训练）
            self.new_model = torch.load(self.mode_path)

        self.new_model.eval()  # 模型转为test模式、

    def transforms(self, img_data):
        img_data = img_data.astype(np.float32)
        img_data = (img_data / 255.0 - self.mean) / self.std  # img.shape = (30,63,3)
        img_data = np.clip(img_data, -1, 1)
        img_data = img_data.transpose([2, 0, 1])  # img.shape = (3,30,63)
        img_data = torch.FloatTensor(img_data)
        return img_data

    def predict(self):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 构建网络架构
        # new_model = models.resnet18(pretrained=False)
        # num_ftrs = new_model.fc.in_features  # 获取模型最后一层的输出
        # new_model.fc = nn.Linear(num_ftrs, 5)  # 更改模型最后一层，输出类别为5
        #
        # # 加载模型(GPU或CPU模型)
        # model = torch.load(self.mode_path, map_location=lambda storage, loc: storage)
        # new_model.load_state_dict(model.state_dict())
        # new_model = new_model.to(device)
        # new_model.eval()  # 模型转为test模式、

        # print(new_model)

        img_list = glob.glob(os.path.join(self.file_path, '*'))

        rights = []
        lables = []
        for img_file in img_list:
            img_data = np.asarray(Image.open(img_file).resize(self.size))
            img_data = self.transforms(img_data)

            # print(img_data.size())
            # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
            img_data = img_data.unsqueeze(0)

            # print(img_data.size())
            if torch.cuda.is_available():
                img_data = img_data.cuda()

            predicted = self.new_model(img_data)

            if torch.cuda.is_available:
                predicted = predicted.cpu()

            if platform.system() == 'Windows':
                lable = torch.LongTensor([int(img_file.split('\\')[-1].split('_')[0]) - 1])
            else:
                lable = torch.LongTensor([int(img_file.split('/')[-1].split('_')[0]) - 1])
            lables.append(lable.data.numpy()[0])

            right = rightness(predicted, lable)
            rights.append(right)

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("准确率：{:.3f},识别个数：{}".format(right_ratio, len(lables)))
        # print(np.array(lables))
        # print(np.array([i[3] for i in rights]).flatten())
        if platform.system() == 'Windows':
            confusionMatrix(np.array(lables), np.array([i[3] for i in rights]).flatten(), self.is_migrate)


if __name__ == '__main__':

    # file_path = 'src/test_action_data.npy'
    if platform.system() == 'Windows':
        file_path = 'D:/home/developer/TrainData/actionImage/test'
    else:
        file_path = '/home/yanjilong/DataSets/actionImage/test'

    with Timer() as t:
        predict = Predict(file_path, is_migrate=False)
        predict.predict()
    # print('Testing complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    print('predict time {0}'.format(str(t.interval)[:5]))
