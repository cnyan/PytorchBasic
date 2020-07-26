# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/24 22:13
@Describe：

"""
import itertools

import torch
from re_cnn_data_input import Input_Data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 通过混淆矩阵，分析错误的分类
from sklearn.metrics import confusion_matrix
import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class Cnn_Predict():
    def __init__(self):
        self.model = torch.load('src/re_cnn_model.pkl')
        self.model.eval()
        self.test_action_data_set = Input_Data('test_action_data.npy')
        print(f'test_data size:{len(self.test_action_data_set)}')

    def predict(self):
        rights = []
        labels = []
        for data, label in self.test_action_data_set:
            data = data.unsqueeze(0)  # 扩展一个维度
            label = torch.LongTensor([int(label)])

            labels.append(label)
            output = self.model(data)

            right = self.rightness(output, label)
            rights.append(right)

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("准确率：{:.3f},识别个数：{}".format(right_ratio, len(labels)))
        self.plot_confusion_matrix(np.array(labels), np.array([i[3] for i in rights]).flatten(),
                                   classes=[0, 1, 2, 3, 4])

    # 自定义计算准确度的函数
    def rightness(self, predict, label):
        '''
        计算准确度
        :param predict:
        :param label:
        :return: right,len(label),score,pred_idx
        '''
        prob = F.softmax(predict, dim=1)
        score = torch.max(prob, 1)[0].data.numpy()[0]
        pred_idx = torch.max(predict, 1)[1]  # 最大值下标
        right = pred_idx.eq(label.data.view_as(pred_idx)).sum()  # 返回 0（false)，1(true)
        return right.data.item(), len(label), score, pred_idx.data.numpy()[0]

    # 混淆矩阵
    def plot_confusion_matrix(self, y_label, y_predict, classes,
                              title='Confusion matrix'):

        cm = confusion_matrix(y_label, y_predict)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45)  # 参数： rotation=45，label倾斜45°
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.ylim(len(cm) - 0.5, -0.5)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('plt_img/re_cnn_predict.jpg')
        plt.show()


if __name__ == '__main__':
    cnn_predict = Cnn_Predict()
    with Timer() as t:
        cnn_predict.predict()
    print('predict time {0}'.format(str(t.interval)[:5]))
