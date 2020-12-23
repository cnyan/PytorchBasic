# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/23 18:34
@Describe：

"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools


def plot_confusion_matrix(y_label, y_predict, classes, savePath,
                          title='Confusion matrix'):
    """
    绘制混淆矩阵
    :param y_predict:
    :param classes:
    :param savePath:
    :param title:
    :return:
    """
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
    plt.savefig(savePath)
    plt.show()


def metrics(ytest, y_predict):
    """
    结果分析
    :param y_predict:
    :return:
    """
    # 预测精度分析
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # 准确度
    accuracy = accuracy_score(ytest, y_predict)
    f1 = f1_score(ytest, y_predict, average='macro')
    # 精确度
    precision = precision_score(ytest, y_predict, average='macro')
    # 召回率： 表示预测的正样例占全部正样例的比例
    recall = recall_score(ytest, y_predict, average='macro')

    print('动作分类预测准确度：%s' % accuracy)
    print('动作分类预测精确度：%s' % precision)
    print('动作分类预测F1：%s' % f1)
    print('动作分类召回率：%s' % recall)