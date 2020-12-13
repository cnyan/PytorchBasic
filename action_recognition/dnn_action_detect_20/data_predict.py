# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/8/27 21:17
@Describe：

"""
from data_input import read_csv
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dnn_action_nn import Action_dnn, Action_dnn_regular
import torch
from torch.autograd import Variable
import math
from torch import nn
import itertools



class dnn_action_detect(nn.Module):
    def __init__(self):
        super(dnn_action_detect, self).__init__()
        self.layer1 = torch.nn.Linear(20, 400)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(400, 400)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Linear(400, 2)
        self.layer6 = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

class utils():
    def plot_confusion_matrix(y_label, y_predict, classes, savePath,
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
        plt.savefig(savePath)
        plt.show()

    def metrics(ytest, y_predict):
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


class Dnn_predict():
    def __init__(self):
        # self.model = torch.load('src/model/dnn-pca_model.pkl')
        self.model = dnn_action_detect()
        self.model.load_state_dict(torch.load('src/dnn_action_model.pkl'))
        self.model.eval()  # 指定测试

    def predict(self):
        # 加载测试数据集
        df_datas, df_labels = read_csv().get_test_data()

        rights = []
        pred_label = []
        pred_score = []
        for i, data in enumerate(zip(df_datas, df_labels)):
            x, y = data
            x = torch.FloatTensor(x).view(1, -1)
            y = torch.LongTensor(y)

            x = Variable(x)
            y = Variable(y)

            predict = self.model(x)

            right = self.rightness(predict, y)
            rights.append(right)
            pred_label.append(torch.max(predict, 1)[1].cpu().data.numpy()[0])
            pred_score.append(right[2])

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("准确率：{:.3f},识别个数：{}".format(right_ratio, len(df_labels.flatten())))

        utils.plot_confusion_matrix(np.array(df_labels).flatten(), np.array(pred_label), [0, 1, 2, 3, 4],
                                    'plt_img/dnn_pca_predict.jpg',
                                    title='DNN-PCA Confusion matrix')
        utils.metrics(np.array(df_labels).flatten(), np.array(pred_label))

    # 自定义计算准确度的函数
    def rightness(self, predictions, labels):
        pred = torch.max(predictions, 1)[1]  # 最大值下标
        rights = pred.eq(labels.data.view_as(pred)).sum()
        score = np.max([math.pow(math.e, i) for i in predictions.cpu().detach().numpy().flatten()])
        return rights.cpu(), len(labels), score

if __name__ == '__main__':
    dnn_predict = Dnn_predict()
    dnn_predict.predict()
