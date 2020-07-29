# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/7/24 17:52
@Describe：
        数据训练
"""

from re_cnn_nn import Action_Net_CNN
from re_cnn_data_input import Input_Data
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
from torch.autograd import Variable
import time
import os
import matplotlib.pyplot as plt


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class Cnn_train():
    def __init__(self):
        self.action_data_train_set = Input_Data("train_action_data.npy")
        self.action_data_valid_set = Input_Data('valid_action_data.npy')
        print(f'train_data size:{len(self.action_data_train_set)}')
        print(f'valid_data size:{len(self.action_data_valid_set)}')

        # 按批加载 pyTorch张量
        self.action_train_data_gen = DataLoader(self.action_data_train_set, batch_size=128, shuffle=True,
                                                num_workers=1)  # 分成数组（len/128）个batch，每个batch长度是128
        self.action_valid_data_gen = DataLoader(self.action_data_valid_set, batch_size=128, shuffle=True,
                                                num_workers=1)  # 分成数组（len/128）个batch，每个batch长度是128
        self.model = Action_Net_CNN()

    def train(self):
        since = time.time()

        model_ft = self.model

        if torch.cuda.is_available():
            model_ft = model_ft.cuda()  # 告知 pyTorch 在Gpu上运行

        dataset_sizes = {'train': len(self.action_train_data_gen.dataset),
                         'valid': len(self.action_valid_data_gen.dataset)}
        dataloaders = {'train': self.action_train_data_gen, 'valid': self.action_valid_data_gen}

        # 构建模型:损失函数和优化模型
        num_epochs = 200
        criterion = nn.CrossEntropyLoss()  # criterion:惩罚规则-- 损失函数
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=0.01)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        train_loss = []
        valid_loss = []
        right_ratio = []  # 正确率

        for epoch in range(num_epochs):
            print('-' * 30)
            print('Epoch {}/{} '.format(epoch, num_epochs - 1))
            print(f"the lr is :{optimizer_ft.param_groups[0]['lr']}")

            # 每轮都有训练和验证过程
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model_ft.train(True)
                else:
                    model_ft.eval()

                running_loss = 0.0
                running_corrects = 0  # correct 修正，改正

                for data in dataloaders[phase]:
                    inputs, labels = data  # 获取输入
                    # 封装成变量
                    if torch.cuda.is_available():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # 梯度参数清0
                    optimizer_ft.zero_grad()
                    # 前向算法
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    # print(labels)
                    # print('*'*20)
                    # print(preds)
                    # print('*' * 40)
                    loss = criterion(outputs, labels)  # 损失函数

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

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 深度复制模型
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model_ft.state_dict()

                # if phase == 'train':
                #     exp_lr_scheduler.step()

        time_elapsed = time.time() - since
        print('-' * 30)
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model_ft.load_state_dict(best_model_wts)

        if os.path.exists('src/re_cnn_model.pkl'):
            os.remove('src/re_cnn_model.pkl')
        torch.save(model_ft, 'src/re_cnn_model.pkl')

        self.plt_image(train_loss, valid_loss, right_ratio)

    def plt_image(self, train_loss, valid_loss, right_ratio):
        plt.plot(train_loss, label='Train Loss')
        plt.plot(valid_loss, label='Valid Loss')
        plt.plot(right_ratio, label='Valid Accuracy')
        plt.xlabel('Steps')
        plt.ylabel('Loss & Accuracy')
        plt.legend()
        plt.savefig("plt_img/cnn_train_loss.png")
        plt.show()


if __name__ == '__main__':
    cnn_train = Cnn_train()
    cnn_train.train()
