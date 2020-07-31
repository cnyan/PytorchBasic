# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/9/29 10:02
@Describe：
    读取图片，并转换为张量

     需要指出的是，使用GPU训练出来的模型，如果后续用于分类的时候使用的是CPU，则需要将模型进行转化，否则会报错：
    torch.load('tensors.pt', map_location='cpu')
"""
import platform

import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, optim

from torchvision import models

from cnn_action_nn import ActionsDataSet, Action_Net_CNN
from cnn_base_var import action_window_row, action_window_col
import time


# 指定第二块GPU
# torch.cuda.set_device(1)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5, is_migrate=False):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    trin_loss = []
    valid_loss = []
    right_ratio = []  # 正确率

    for epoch in range(num_epochs):
        print('Epoch {}/{} '.format(epoch, num_epochs - 1))
        print('-' * 20)

        # 每轮都有训练和验证过程
        for phase in ['train', 'valid']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # 模型设为训练模式
            else:
                model.train(False)  # 模型设为评估模式

            running_loss = 0.0
            running_corrects = 0  # correct 修正，改正

            # print('*******************' + str(len(dataloaders[phase])))

            # 在数据上进行迭代
            for data in dataloaders[phase]:
                inputs, labels, name = data  # 获取输入
                # 封装成变量
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # print(inputs.size())  # torch.Size([64, 3, 30, 63])

                # 梯度参数清0
                optimizer.zero_grad()

                # 前向算法
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # 只在训练阶段反向优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 统计
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])  # dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            # 计算损失率
            if phase == 'train':
                trin_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)
                right_ratio.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            if phase == 'train':
                scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 绘制误差曲线
    if is_migrate:
        jpg_name = 'plt_img/resnet18_train_loss.png'
    else:
        jpg_name = 'plt_img/cnn_train_loss.png'

    plt.plot(trin_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.plot(right_ratio, label='Valid Accuracy')
    plt.xlabel('Steps')
    plt.ylabel('Loss & Accuracy')
    plt.legend()
    plt.savefig(jpg_name)
    plt.show()

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def imshow(inp):
    """show image for tensor
        对张量再次变形，并且将值 反  归一化
    """
    # inp = inp.numpy().transpose((1, 2, 0))  # transpose 按照指定维度旋转矩阵
    inp = inp.transpose((1, 2, 0))  # transpose 按照指定维度旋转矩阵
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # 反序列化
    inp = np.clip(inp, -1, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    if platform.system() == 'Windows':
        action_train_path = 'D:/home/developer/TrainData/actionImage/train'
        action_valid_path = 'D:/home/developer/TrainData/actionImage/valid'
    else:
        action_train_path = '/home/yanjilong/DataSets/actionImage/train'
        action_valid_path = '/home/yanjilong/DataSets/actionImage/valid'

    # 是否构建迁移网络
    is_migrated = False

    action_data_train_set = ActionsDataSet(os.path.join(action_train_path, '*/'))
    action_data_valid_set = ActionsDataSet(os.path.join(action_valid_path, '*/'))
    print("训练数据集数目是：" + str(len(action_data_train_set)))
    print("验证数据集数目是：" + str(len(action_data_valid_set)))

    print(np.array(action_data_train_set[50][0]).shape)
    # imshow(action_data_train_set[50][0])

    # 按批加载 pyTorch张量
    action_train_data_gen = DataLoader(action_data_train_set, batch_size=64, shuffle=True,
                                       num_workers=1)  # 分成数组75（len/64）个batch，每个batch长度是64
    action_valid_data_gen = DataLoader(action_data_valid_set, batch_size=64, shuffle=True,
                                       num_workers=1)  # 分成数组15（len/64）个batch，每个batch长度是64

    print('训练数据的批次{}'.format(len(action_train_data_gen.dataset)))
    print('验证数据的批次{}'.format(len(action_valid_data_gen.dataset)))

    dataset_sizes = {'train': len(action_train_data_gen.dataset), 'valid': len(action_valid_data_gen.dataset)}
    dataloaders = {'train': action_train_data_gen, 'valid': action_valid_data_gen}

    # ===================================================================================================
    if is_migrated:
        # 构建网络架构(网络迁移)
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features  # 获取模型最后一层的输出
        model_ft.fc = nn.Linear(num_ftrs, 5)  # 更改模型最后一层，输出类别为5
        # print(model_ft)
    else:
        # 自定义网络架构
        model_ft = Action_Net_CNN()
    # ===================================================================================================

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()  # 告知 pyTorch 在Gpu上运行

    # 构建模型:损失函数和优化模型
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()  # criterion:惩罚规则-- 损失函数
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 开始训练
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=20, is_migrate=is_migrated)

    if is_migrated:
        if os.path.exists('src/cnn_model_migrate.pkl'):
            os.remove('src/cnn_model_migrate.pkl')
        torch.save(model_ft, 'src/cnn_model_migrate.pkl')
    else:
        if os.path.exists('src/cnn_model.pkl'):
            os.remove('src/cnn_model.pkl')
        torch.save(model_ft, 'src/cnn_model.pkl')

"""    
    if platform.system() == 'Windows':
        action_path = 'D:/home/developer/TrainData/actionImage/train'
    else:
        action_path = '/home/yanjilong/DataSets/actionImage/train'

    action_data_set = ActionsDataSet(os.path.join(action_path, '*/'))

    # action_path = 'D:\\home\\developer\\TrainData\\actionImage\\allImage'

    actions_one_path = glob.glob(os.path.join(action_path, '1/') + '*.jpg')
    action_one_imgs = np.array(
        [np.array(Image.open(act).resize((action_window_col, action_window_row))) for act in actions_one_path[:64]])
    # action_one_imgs = np.array([np.array(Image.open(act)) for act in actions_one_path[:64]])
    print(len(actions_one_path))
    print(action_one_imgs[0].shape)

    action_one_imgs = action_one_imgs.reshape(-1, action_window_row, action_window_col, 3)

    action_tensor = torch.from_numpy(action_one_imgs)  # torch.Size([64, 30, 63, 3])

    print(action_tensor.size())
    print(action_tensor[0].size())

    print(action_tensor[0].max())
    print(action_tensor[0])

    plt.imshow(action_tensor[0])
    plt.show()
"""
