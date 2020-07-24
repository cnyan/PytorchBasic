# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/8/27 18:48
@Describe：

"""
import os
import sys

import time
import platform
import numpy
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

import torchvision
from torchvision import transforms  # 预处理类
from torchvision.datasets import ImageFolder  # 加载图片
from torchvision import models
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

if platform.system() == 'Windows':
    from kaggle.dogsAndCats.Log import Logger
else:
    from Log import Logger

sys.stdout = Logger()


# 把数据加载到 pyTorch 张量中:ImageFolder


def imshow(inp):
    """show image for tensor
        对张量再次变形，并且将值 反  归一化
    """
    inp = inp.numpy().transpose((1, 2, 0))  # transpose 按照指定维度旋转矩阵
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # 反序列化
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{} '.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每轮都有训练和验证过程
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 模型设为训练模式
            else:
                model.train(False)  # 模型设为评估模式

            running_loss = 0.0
            running_corrects = 0  # correct 修正，改正

            print('*******************' + str(len(dataloaders[phase])))
            # 在数据上进行迭代
            for data in dataloaders[phase]:
                inputs, labels = data  # 获取输入
                # 封装成变量
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    path = ''
    if platform.system() == 'Windows':
        path = 'D:/home/developer/TrainData/dogsandcats'
    else:
        path = '/home/yanjilong/DataSets/dogsandcats'

    if torch.cuda.is_available():
        print('GPU 执行')

    # img_data = Image.open(os.path.join(path, 'train/dog/dog.0.jpg'))
    # print(np.array(img_data))

    '''
    1. 调整数据大小 256*256
    2. 转换pyTorch 张量
    3. 归一化操作
    '''
    simple_transform = transforms.Compose(
        [transforms.Scale((224, 224)), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train = ImageFolder(os.path.join(path, 'train'), simple_transform)
    valid = ImageFolder(os.path.join(path, 'valid'), simple_transform)

    print(len(train))
    print(np.array(train[50][0]).shape)
    imshow(train[50][0])
    # print(train.class_to_idx)  # {'cat': 0, 'dog': 1}
    # print(train.classes)  # ['cat', 'dog']

    # 按批加载 pyTorch张量
    train_data_gen = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=1)
    valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=24)
    dataset_sizes = {'train': len(train_data_gen.dataset), 'valid': len(valid_data_gen.dataset)}
    dataloaders = {'train': train_data_gen, 'valid': valid_data_gen}

    # print('*********' * 8)
    # for datas, labs in train_data_gen:
    #     print(datas.size())  # torch.Size([64, 3, 224, 224])
    #     print(datas[0])


    # 构建网络架构
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features  # 获取模型最后一层的输出
    model_ft.fc = nn.Linear(num_ftrs, 2)  # 更改模型最后一层，输出类别为2
    # print(model_ft)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()  # 告知 pyTorch 在Gpu上运行

    # 构建模型:损失函数和优化模型
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()  # criterion:惩罚规则-- 损失函数
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 开始训练
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)

    print('\n\n')
