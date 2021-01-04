# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/30 20:24
@Describe：

"""
import os
import copy
import numpy as np
import time
import platform
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataToTorch import ActionDataSets
import AUtils
import datetime

import warnings

warnings.filterwarnings('ignore')


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class NN_train():
    def __init__(self, modelNet, model_name, axis='9axis'):
        super(NN_train, self).__init__()
        self.model_name = model_name
        self.axis = axis

        self.action_data_train_set = ActionDataSets('train', axis)
        self.action_data_valid_set = ActionDataSets('valid', axis)
        print(f'train_data shape: {(self.action_data_train_set.data_shape())}')
        print(f'valid_data shape: {(self.action_data_valid_set.data_shape())}')

        # 按批加载 pyTorch张量
        self.action_train_data_gen = DataLoader(self.action_data_train_set, batch_size=64, shuffle=True,
                                                num_workers=2)  # 分成数组（len/128）个batch，每个batch长度是128
        self.action_valid_data_gen = DataLoader(self.action_data_valid_set, batch_size=64, shuffle=True,
                                                num_workers=2)  # 分成数组（len/128）个batch，每个batch长度是128
        self.model = copy.deepcopy(modelNet)

    def train(self):
        since = time.time()

        model_ft = self.model

        if torch.cuda.is_available():
            model_ft = model_ft.cuda()  # 告知 pyTorch 在Gpu上运行

        dataset_sizes = {'train': len(self.action_train_data_gen.dataset),
                         'valid': len(self.action_valid_data_gen.dataset)}
        dataloaders = {'train': self.action_train_data_gen, 'valid': self.action_valid_data_gen}

        # 构建模型:损失函数和优化模型
        num_epochs = 60
        criterion = nn.CrossEntropyLoss()  # criterion:惩罚规则-- 损失函数
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01, weight_decay=0.10)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

        # best_model_wts = self.model.state_dict()
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_acc = 0.0
        train_loss = []
        valid_loss = []
        right_ratio = []  # 正确率

        for epoch in range(1, num_epochs + 1):
            if epoch % 10 == 0:
                print('-' * 30)
                print('{}-{},Epoch {}/{} '.format(self.model_name, self.axis, epoch, num_epochs))
                print(f"the lr is :{optimizer_ft.param_groups[0]['lr']}")

            # 每轮都有训练和验证过程
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model_ft.train(True)
                else:
                    model_ft.eval()

                running_loss = 0.0
                running_corrects = 0  # correct 修正，改正

                for i, data in enumerate(dataloaders[phase]):
                    inputs, labels = data  # 获取输入

                    # 封装成变量
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    else:
                        inputs, labels = inputs, labels

                    # 前向算法
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)  # 损失函数
                    # 梯度参数清0
                    optimizer_ft.zero_grad()

                    # 只在训练阶段反向优化
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()
                    # 统计
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase])  # dataset_sizes[phase]
                epoch_acc = running_corrects.item() / dataset_sizes[phase]

                # 计算损失率
                if phase == 'train':
                    train_loss.append(epoch_loss)
                else:
                    valid_loss.append(epoch_loss)
                    right_ratio.append(epoch_acc)
                if epoch % 10 == 0:
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                # 深度复制模型
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # best_model_wts = model_ft.state_dict()
                    best_model_wts = copy.deepcopy(model_ft.state_dict())

                if phase == 'train':
                    exp_lr_scheduler.step()

        time_elapsed = time.time() - since
        print('-' * 30)
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        # model_ft.load_state_dict(best_model_wts)

        if os.path.exists(f'src/model/{self.model_name}_{self.axis}_model.pkl'):
            os.remove(f'src/model/{self.model_name}_{self.axis}_model.pkl')
        torch.save(model_ft.state_dict(), f'src/model/{self.model_name}_{self.axis}_model.pkl')
        self.plt_image(train_loss, valid_loss, right_ratio)

    def plt_image(self, train_loss, valid_loss, right_ratio):
        plt.title(f'{self.model_name}_{self.axis} training')
        plt.plot(train_loss, label='Train Loss')
        plt.plot(valid_loss, label='Valid Loss')
        plt.plot(right_ratio, label='Valid Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('Loss & Accuracy')
        plt.legend()
        plt.savefig(f"src/plt_img/{self.model_name}_{self.axis}_train_loss.png")
        plt.show()
        plt.close()

    def showImg(self, data):
        # plt.imshow(data)
        # plt.show()
        # 可视化图像
        plt.figure(figsize=(3, 3))
        plt.imshow(data, cmap=plt.cm.gray)
        plt.axis('on')  # 不显示坐标
        plt.show()


class NN_Predict():
    def __init__(self, modelNet, model_name, axis):
        super(NN_Predict, self).__init__()
        self.model = modelNet
        self.axis = axis

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{axis}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.model_name = model_name

        if platform.system() == 'Windows':
            test_dir = fr'D:/home/DataRec/actionImage/{axis}/test'
        else:
            test_dir = fr'/home/yanjilong/dataSets/DataRec/actionImage/{axis}/test'

        self.test_action_data_set = DataLoader()
        print(f'test_data size:{len(self.test_action_data_set)}')

    def predict(self):
        rights = []
        labels = []
        for data, label in self.test_action_data_set:
            data = data.unsqueeze(0)  # 扩展一个维度
            label = torch.LongTensor([int(label)])
            if torch.cuda.is_available():
                data = data.cuda()

            labels.append(label)
            output = self.model(data)

            right = self.rightness(output, label)
            rights.append(right)

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("模式{}-{},准确率：{:.3f},识别个数：{}".format(self.model_name, self.axis, right_ratio, len(labels)))

        AUtils.plot_confusion_matrix(np.array(labels), np.array([i[3] for i in rights]).flatten(),
                                     classes=[0, 1, 2, 3, 4],
                                     savePath=f'src/plt_img/{self.model_name}_{self.axis}_predict.png',
                                     title=f'{self.model_name}_{self.axis}_predict')

    # 自定义计算准确度的函数
    def rightness(self, predict, label):
        '''
        计算准确度
        :param predict:
        :param label:
        :return: right,len(label),score,pred_idx
        '''
        prob = F.softmax(predict, dim=1)
        score = torch.max(prob, 1)[0].cpu().data.numpy()[0]
        pred_idx = torch.max(predict, 1)[1].cpu()  # 最大值下标(类别)
        right = pred_idx.eq(label.data.view_as(pred_idx)).sum()  # 返回 0（false)，1(true)
        return right.data.item(), len(label), score, pred_idx.cpu().data.numpy()[0]


if __name__ == '__main__':
    """
    窗口长度 36
    xyz 6 (14, 36, 3)
    xyz 9 (21, 36, 3)
    awh 9 (21, 36, 3)
    org 6 (42, 36, 3)
    org 9 (63, 36, 3)
    """
    from AUtils import make_print_to_file  # 打印日志
    from d_Single_NN_Net import MyDnnNet

    make_print_to_file()
    axis_all = ['9axis', '6axis']

    for axis in axis_all:
        myDnnNet = MyDnnNet(7 * int(axis[0]) * 36)
        models_all = {'myDnnNet': myDnnNet}

        for model_name, model in models_all.items():
            print('===================********begin begin begin*********=================')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            nn_train = NN_train(model, model_name, axis=axis)
            with Timer() as t:
                nn_train.train()
            print('training time {0}'.format(str(t.interval)[:5]))

            nn_predict = NN_Predict(model, model_name, axis=axis)
            with Timer() as t:
                nn_predict.predict()
            print('predict time {0}'.format(str(t.interval)[:5]))
