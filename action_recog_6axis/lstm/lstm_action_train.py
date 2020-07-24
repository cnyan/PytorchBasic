# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/12/24 11:40
@Describe：

"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lstm_action_nn import Action_Net_LSTM, ActionsDataSet
import platform
import os
import numpy as np

import matplotlib.pyplot as plt
# 预测精度分析
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Hyper Parameters
# （每张图片输入n个step，n表示图片高度，每个step输入一行像素信息）
EPOCH = 20  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 40  # lstm time step / image height
INPUT_SIZE = 28  # lstm input size / image width
LR = 0.01  # learning rate


def get_score(predict, y):
    predict = np.around(predict, decimals=1)
    predict_y = np.array(predict).argmax(axis=1)
    predict_value = np.array(predict).max(axis=1)
    # print(predict)
    # print(predict_y)
    # print(predict_value)

    # 准确度
    accuracy = accuracy_score(y, np.array(predict_y))
    return accuracy


if __name__ == '__main__':
    if platform.system() == 'Windows':

        action_train_path = 'D:/home/developer/TrainData/actionImage/train'
        action_valid_path = 'D:/home/developer/TrainData/actionImage/valid'
    else:
        action_train_path = '/home/yanjilong/DataSets/actionImage/train'
        action_valid_path = '/home/yanjilong/DataSets/actionImage/valid'

    action_data_train_set = ActionsDataSet(os.path.join(action_train_path, '*/'))
    action_data_valid_set = ActionsDataSet(os.path.join(action_valid_path, '*/'))
    print("训练数据集数目是：" + str(len(action_data_train_set)))
    print("验证数据集数目是：" + str(len(action_data_valid_set)))


    # 按批加载 pyTorch张量
    action_train_data_gen = DataLoader(action_data_train_set, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=1)  # 分成数组75（len/64）个batch，每个batch长度是64
    action_valid_data_gen = DataLoader(action_data_valid_set, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=1)  # 分成数组15（len/64）个batch，每个batch长度是64

    lstm = Action_Net_LSTM()
    if torch.cuda.is_available():
        lstm = lstm.cuda()

    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    all_train_loss = []
    all_valid_loss = []
    all_valid_accuracy = []
    for epoch in range(50):

        train_loss_list = []
        valid_loss_list = []
        valid_accuracy_list = []
        for step, (b_x, b_y, b_name) in enumerate(action_train_data_gen):  # gives batch data
            b_x = b_x.type(torch.FloatTensor)
            b_x = b_x.view(-1, 40, 63)

            # 封装成变量
            if torch.cuda.is_available():
                b_x = Variable(b_x.cuda())
                b_y = Variable(b_y.cuda())
            else:
                b_x, b_y = Variable(b_x), Variable(b_y)
            lstm.train()
            output = lstm(b_x)  # rnn output

            loss = loss_func(output, b_y)  # cross entropy loss

            optimizer.zero_grad()  # clear gradients for this training step

            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            train_loss_list.append(loss.cpu().detach())

        for step, (b_x, b_y, b_name) in enumerate(action_valid_data_gen):  # gives batch data

            b_x = b_x.type(torch.FloatTensor)
            b_x = b_x.view(-1, 40, 63)

            # 封装成变量
            if torch.cuda.is_available():
                b_x = Variable(b_x.cuda())
                b_y = Variable(b_y.cuda())
            else:
                b_x, b_y = Variable(b_x), Variable(b_y)

            lstm.eval()
            output = lstm(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss

            valid_loss_list.append(loss.cpu().detach())

            predict_label = F.softmax(output.cpu()).detach().numpy()
            valid_accuracy_list.append(get_score(predict_label, b_y.cpu().detach().numpy()))

        print(f'{epoch} epoch, train loss value is {np.array(train_loss_list).flatten().mean()}')
        print(f'{epoch} epoch, valid loss value is {np.array(valid_loss_list).flatten().mean()}')
        print(f'{epoch} epoch, valid accuracy loss value is {np.array(valid_accuracy_list).flatten().mean()}')

        all_train_loss.append(float(str(np.array(train_loss_list).flatten().mean())[0:4]))
        all_valid_loss.append(float(str(np.array(valid_loss_list).flatten().mean())[0:4]))
        all_valid_accuracy.append(float(str(np.array(valid_accuracy_list).flatten().mean())[0:4]))

    if os.path.exists('src/lstm_model.pkl'):
        os.remove('src/lstm_model.pkl')

    torch.save(lstm, 'src/lstm_model.pkl')

    plt.plot(np.array(all_train_loss), label='train loss')
    plt.plot(np.array(all_valid_loss), label='valid loss')
    plt.plot(np.array(all_valid_accuracy), label='valid accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Loss & Accuracy')
    plt.legend()
    plt.savefig('plt_img/lstm_train_loss.jpg')
    plt.show()

