# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/10/12 18:39
@Describe：

"""
from dnn_data_input import read_data_sets

import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import math
import platform
import os
from dnn_base_var import action_window_col, action_window_row
from dnn_action_nn import Action_dnn, Action_dnn_regular

if not platform.system() == 'Windows':
    plt.switch_backend('agg')


# 自定义计算准确度的函数
def rightness(predictions, labels):
    pred = torch.max(predictions, 1)[1]  # 最大值下标
    rights = pred.eq(labels.data.view_as(pred)).sum()
    # print(f'{pred}:{rights}:{len(labels)}')

    return rights.cpu(), len(labels)


if __name__ == '__main__':
    # 数据处理
    actions_sets = read_data_sets('src/train_action_data.npy', valid_size=0.2)
    train_actions_data = actions_sets['train_data']
    valid_actions_data = actions_sets['validation_data']

    train_actions_label = actions_sets['train_label']
    valid_actions_label = actions_sets['validation_label']

    batch_size = 128

    # model = Action_dnn()
    model = Action_dnn_regular()

    if torch.cuda.is_available():
        model = model.cuda()

    # 开始训练
    cost = torch.nn.NLLLoss()  # 交叉熵损失函数
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # 优化函数,加入L2正则化
    records = []

    # 20次循环
    losses = []
    for epoch in range(100):
        for start in range(0, len(train_actions_data), batch_size):
            end = start + batch_size if start + batch_size < len(train_actions_data) else len(train_actions_data)
            x = torch.FloatTensor(train_actions_data[start:end])
            y = torch.LongTensor(train_actions_label[start:end].flatten().astype(np.int64))
            # 模型设为训练模式
            model.train(True)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

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

        # 每隔50，计算一下验证集
        if epoch % 10 == 0:
            val_losses = []
            rights = []
            for j, val in enumerate(zip(valid_actions_data, valid_actions_label)):
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


    if os.path.exists('src/dnn_model.pkl'):
        os.remove('src/dnn_model.pkl')

    # 保存训练后的模型
    torch .save(model, 'src/dnn_model.pkl')

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
    # plt.show()