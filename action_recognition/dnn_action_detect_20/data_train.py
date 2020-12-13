#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from data_input import read_csv
import torch
from torch import nn
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import math
import platform
import joblib
import os

#
# class dnn_action_detect(nn.Module):
#     def __init__(self):
#         super(dnn_action_detect, self).__init__()
#         self.layer1 = torch.nn.Linear(20, 400)
#         self.layer2 = torch.nn.ReLU()
#         self.layer3 = torch.nn.Linear(400, 400)
#         self.layer4 = torch.nn.ReLU()
#         self.layer5 = torch.nn.Linear(400, 400)
#         self.layer6 = torch.nn.ReLU()
#         self.layer7 = torch.nn.Linear(400, 2)
#         self.layer8 = torch.nn.LogSoftmax(dim=1)
#
#     def forward(self, input):
#         out = self.layer1(input)
#         out = F.dropout2d(out, p=0.5, training=self.training)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.dropout2d(out, p=0.5, training=self.training)
#         out = self.layer4(out)
#         out = self.layer5(out)
#         out = F.dropout2d(out, p=0.5, training=self.training)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         out = self.layer8(out)
#         return out


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

# 自定义计算准确度的函数
def rightness(predictions, labels):
    pred = torch.max(predictions, 1)[1]  # 最大值下标
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights.cpu(), len(labels)


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = read_csv().split_data()

    model = dnn_action_detect()

    # 开始训练
    cost = torch.nn.NLLLoss()  # 交叉熵损失函数
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001,momentum=0.9)  # 优化函数,加入L2正则化
    records = []

    batch_size = 100
    # 20次循环
    losses = []
    for epoch in range(200):
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size if start + batch_size < len(x_train) else len(x_train)
            x = torch.FloatTensor(x_train[start:end])
            y = torch.LongTensor(y_train[start:end].flatten().astype(np.int64))
            # 模型设为训练模式
            model.train(True)

            x = Variable(x, requires_grad=True)
            y = Variable(y)

            # 模型预测
            predict = model(x)
            # print(f'{torch.max(predict, 1)[1].data.numpy()}:{y.data}')
            # 计算损失函数
            loss = cost(predict, y)
            losses.append(loss.cpu().data.numpy())

            # 清空梯度
            optimizer.zero_grad()
            # 反向梯度计算
            loss.backward()
            # 优化
            optimizer.step()

        # 每隔500，计算一下验证集
        if epoch % 10 == 0:
            val_losses = []
            rights = []
            for j, val in enumerate(zip(x_valid, y_valid)):
                x, y = val
                x = torch.FloatTensor(x).view(1, -1)
                y = torch.LongTensor(y)
                # 模型设为预测模式
                model.train(False)
                model.eval()
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                x = Variable(x, requires_grad=True)
                y = Variable(y)

                predict = model(x)
                # print(predict)
                # print(f'{torch.max(predict, 1)[1].data.numpy()}:{y.data}')
                # print([math.pow(math.e, i) for i in predict.cpu().detach().numpy().flatten()])

                right = rightness(predict, y)
                rights.append(right)
                loss = cost(predict, y)
                val_losses.append(loss.cpu().data.numpy())

            # 计算校验集的平均准确度
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                        np.mean(val_losses), right_ratio))
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])

    if os.path.exists('src/dnn_action_model.pkl'):
        os.remove('src/dnn_action_model.pkl')

    torch.save(model.state_dict(), 'src/dnn_action_model.pkl')

    # 绘制误差曲线
    a = [i[0] for i in records]
    b = [i[1] for i in records]
    c = [i[2] for i in records]
    plt.plot(a, label='Train Loss')
    plt.plot(b, label='Valid Loss')
    plt.plot(c, label='Valid Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Loss & Accuracy')
    plt.legend()
    plt.savefig('plt_img/dnn_train_loss.png')
    plt.show()
