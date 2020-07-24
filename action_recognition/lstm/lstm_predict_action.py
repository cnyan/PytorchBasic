# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/12/24 16:27
@Describe：

"""
import numpy as np
import platform
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

import math
import os
import glob
import time
# 通过混淆矩阵，分析错误的分类
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from lstm_base_var import action_window_row, action_window_col

import warnings

warnings.filterwarnings('ignore')


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


def rightness(predictions, labels):
    predictions = predictions.cpu()
    prob = F.softmax(predictions, dim=1)

    score = torch.max(prob, 1)[0].cpu().data

    pred = torch.max(predictions, 1)[1]  # 最大值下标

    rights = pred.eq(labels.data.view_as(pred)).sum()

    # if score < 0.6:
    #     isError = True  # 识别错误
    #     if pred.cpu().data.numpy()[0] == labels.data.numpy()[0]:
    #         isError = False  # 正确
    #     print('pred:{},label:{},score:{:.2f},isError:{}'.format(pred.cpu().data.numpy(), labels.data.numpy(), score,
    #                                                             isError))

    return rights.cpu(), len(labels), score, pred.data.numpy()


# 混淆矩阵
def confusionMatrix(y_test, y_predict):
    sns.set()
    mat = confusion_matrix(y_test, y_predict)
    sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.savefig('plt_img/cm_lstm_predict.jpg')
    plt.show()


class Predict:
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

        self.size = (action_window_col, action_window_row)

        self.mode_path = 'src/lstm_model.pkl'

        # 使用自定义网络（CPU与GPU独自训练）
        self.new_model = torch.load(self.mode_path)

        self.new_model.eval()  # 模型转为test模式、

    def transforms(self, img_data):
        L_image = np.array(Image.fromarray(img_data, mode='RGB').convert('L'))  # 灰度图像（40，63）
        L_image = np.clip(L_image / 255.0, 0, 1)  # float64
        img_data = torch.FloatTensor(L_image)
        return img_data

    def predict(self):
        img_list = glob.glob(os.path.join(self.file_path, '*'))

        rights = []
        lables = []
        for img_file in img_list:
            img_data = np.asarray(Image.open(img_file).resize(self.size))
            img_data = self.transforms(img_data)

            # print(img_data.size())
            # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
            img_data = img_data.unsqueeze(0)

            if torch.cuda.is_available():
                self.new_model = self.new_model.cuda()
                img_data = img_data.cuda()

            predicted = self.new_model(img_data)

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
            confusionMatrix(np.array(lables), np.array([i[3] for i in rights]).flatten())


if __name__ == '__main__':

    # file_path = 'src/test_action_data.npy'
    if platform.system() == 'Windows':
        file_path = 'D:/home/developer/TrainData/actionImage/test'
    else:
        file_path = '/home/yanjilong/DataSets/actionImage/test'

    with Timer() as t:
        predict = Predict(file_path)
        predict.predict()
    # print('Testing complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    print('predict time {0}'.format(str(t.interval)[:5]))
