# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/10/15 10:41
@Describe：

"""

import numpy as np
from sklearn import preprocessing
# 通过混淆矩阵，分析错误的分类
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dnn_base_var import action_window_col, action_window_row
from dnn_action_nn import Action_dnn, Action_dnn_regular
import torch
from torch.autograd import Variable
import math
import joblib

import time
import platform
import warnings

warnings.filterwarnings('ignore')


# 显示数组全部元素
# np.set_printoptions(threshold=np.inf)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


# 自定义计算准确度的函数
def rightness(predictions, labels):
    pred = torch.max(predictions, 1)[1]  # 最大值下标
    rights = pred.eq(labels.data.view_as(pred)).sum()
    # print(f'{pred}:{rights}:{len(labels)}')
    # 反向求概率，取消log_softmax()中log的影响
    score = np.max([math.pow(math.e, i) for i in predictions.cpu().detach().numpy().flatten()])

    # if score < 0.6:
    #     isError = True  # 识别错误
    #     if pred.cpu().data.numpy()[0] == labels.data.numpy()[0]:
    #         isError = False  # 正确
    #     print('pred:{},label:{},score:{:.2f},isError:{}'.format(pred.cpu().data.numpy(), labels.data.numpy(), score,
    #                                                             isError))

    return rights.cpu(), len(labels), score


# 混淆矩阵
def confusionMatrix(y_test, y_predict):
    sns.set()
    y_test = np.array(y_test).flatten()
    y_predict = np.array(y_predict)
    mat = confusion_matrix(y_test, y_predict)
    sns.heatmap(mat, square=True, annot=True, cbar=True)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.savefig('plt_img/dnn_predict.png')
    plt.show()


def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 绘制混淆矩阵
    # ==============================================================
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           # xticklabels=['0', '1', '2', '3', '4', '5'],
           # yticklabels=['0', '1', '2', '3', '4', '5'],
           title="Normalized confusion matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('plt_img/cm.jpg')
    plt.show()


class Predict():
    def __init__(self):
        super().__init__()
        self.file_path = 'src/test_action_data.npy'
        self.mode_path = 'src/dnn_model.pkl'

        self.model = Action_dnn_regular()
        # # 加载模型(GPU或CPU模型)
        if platform.system() == 'Windows':
            # 加载模型(GPU或CPU模型)
            model_weights = torch.load(self.mode_path, map_location=lambda storage, loc: storage)
        else:
            # 加载模型(GPU或CPU模型)
            model_weights = torch.load(self.mode_path)

        self.model.load_state_dict(model_weights.state_dict())

        # self.model = torch.load(self.mode_path)
        self.model.eval()  # 指定测试

    def predict(self):
        # 加载测试数据集
        df_array = np.load(self.file_path)
        df_datas = df_array[:, :-1]
        df_labels = df_array[:, -1:]

        rights = []
        test_label = []
        pred_label = []
        pred_score = []
        for i, data in enumerate(zip(df_datas, df_labels)):
            x, y = data

            test_label.append(y)
              # 正则化数据
            x = torch.FloatTensor(x).view(1, -1)
            y = torch.LongTensor(y)
            # if torch.cuda.is_available():
            #     x = x.cuda()
            #     y = y.cuda()

            x = Variable(x)
            y = Variable(y)

            predict = self.model(x)
            # 最大值下标,即预测出的动作分类
            # action_class = torch.max(predict, 1)[1].cpu().data.numpy()[0]
            # 最大值对应的概率，即动作得分
            # action_score = np.max([math.pow(math.e, i) for i in predict.cpu().detach().numpy().flatten()])

            right = rightness(predict, y)
            rights.append(right)
            pred_label.append(torch.max(predict, 1)[1].cpu().data.numpy()[0])
            pred_score.append(right[2])

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("准确率：{:.3f},识别个数：{}".format(right_ratio, len(df_labels.flatten())))
        if platform.system() == 'Windows':
            confusionMatrix(test_label, pred_label)
            # draw_confusion_matrix(test_label, pred_label)
        # print(df_labels.flatten())
        # print(np.array(pred_label))
        # print(np.array(pred_score))


if __name__ == '__main__':
    # file_path = 'src/test_action_data.npy'

    with Timer() as t:
        predict = Predict()
        predict.predict()

    print('predict time {0}'.format(str(t.interval)[:5]))
