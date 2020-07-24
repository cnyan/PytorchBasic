# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/9/29 15:17
@Describe：

"""
import numpy as np
from PIL import Image
from glob import glob
import os
import matplotlib.pyplot as plt
import platform

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from cnn_base_var import action_window_row, action_window_col


# 行长：action_window_col  列宽：action_window_row
class ActionsDataSet(Dataset):

    def __init__(self, img_data_root, size=(action_window_col, action_window_row)):
        """
        :param img_data_path:
        :param size:  width = 63,height = 30
        """
        super().__init__()
        self.files = glob(img_data_root + '*.jpg')
        self.size = size
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        img_data = np.asarray(Image.open(self.files[item]).resize(self.size))
        if platform.system() == 'Windows':
            img_lable = self.files[item].split('\\')[-1].split('_')[0]
        else:
            img_lable = self.files[item].split('/')[-1].split('_')[0]
        img_name = self.files[item]

        img_data = img_data.astype(np.float32)
        img_data = (img_data / 255.0 - self.mean) / self.std  # img.shape = (30,63,3)
        img_data = np.clip(img_data, -1, 1)
        img_data = img_data.transpose([2, 0, 1])  # img.shape = (3,30,63)

        return img_data, int(img_lable) - 1, img_name

    def show_img(self, item):
        """
        show image with item
        :param item:
        :return:
        """
        img_data, lab, name = self.__getitem__(item)
        img_data = img_data.transpose(1, 2, 0)  # transpose 按照指定维度旋转矩阵
        img_data = self.std * img_data + self.mean  # 反序列化
        img_data = np.clip(img_data, -1, 1)
        plt.imshow(img_data)
        plt.show()


# 自定义的神经网络
class Action_Net_CNN(nn.Module):
    def __init__(self):
        super(Action_Net_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (3,action_window_row,action_window_col)
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),  # (16,action_window_row,action_window_col)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 3)),  # (16,action_window_row/2,action_window_col/3)
        )
        self.conv2 = nn.Sequential(  # (16,action_window_row/2,action_window_col/3)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,action_window_row/2,action_window_col/3)
            nn.ReLU(),  # (32,action_window_row/2,action_window_col/3)
            nn.MaxPool2d(kernel_size=(2, 3)),  # ((32,action_window_row/2/2,action_window_col/3/3)
        )
        self.input_size = int(32 * (action_window_row / 4) * (action_window_col / 9))
        # self.layer1 = nn.Linear(self.input_size, 500)
        # self.layer2 = nn.ReLU()
        # self.layer3 = nn.Linear(500, 100)
        # self.layer4 = nn.ReLU()
        self.out = nn.Linear(self.input_size, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.dropout2d(x, p=0.2, training=self.training)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size,32*7*7),
        # 假设：和torch.unsqueeze(）作用相反
        x = x.view(x.size(0), -1)

        # x = self.layer1(x)
        # x = F.dropout2d(x, p=0.5, training=self.training)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        output = self.out(x)
        return output


def imshow(inp):
    """show image for tensor
        对张量再次变形，并且将值 反  归一化
    """
    inp = inp.numpy().transpose((1, 2, 0))  # transpose 按照指定维度旋转矩阵
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # 反序列化
    inp = np.clip(inp, -1, 1)
    plt.imshow(inp)
    plt.show()


if __name__ == '__main__':
    if platform.system() == 'Windows':
        action_path = 'D:/home/developer/TrainData/actionImage/train'
    else:
        action_path = '/home/yanjilong/DataSets/actionImage/train'

    action_data_set = ActionsDataSet(os.path.join(action_path, '*/'))

    print("数据集数目是：" + str(len(action_data_set)))
    # print(action_data_set.files)
    # action_data_set.show_img(0)
    # print(action_data_set[0])

    # ========================加载整个数据集
    print('=========方法一==============' * 3)
    for img, label, img_file_name in action_data_set:
        # 创建张量
        action_one_imgs = img.reshape(-1, 30, 63, 3)
        action_tensor = torch.from_numpy(action_one_imgs)  # torch.Size([64, 30, 63, 3])
        # print(label)
        # print(action_tensor.size())
        # print(action_tensor[0].size())

    # ===================按批次加载数据
    print('=========方法二==============' * 3)
    action_data_loader = DataLoader(action_data_set, batch_size=32, shuffle=True,
                                    num_workers=1)  # 分成数组27（len/32）个batch，每个batch长度是32

    # for batch in range(10):
    #     print(str(batch) + ":" + str(len(action_data_loader)))
    #     for imgs, labels, img_names in action_data_loader:  # 共（len/32）个batch次循环
    #         print(imgs[0].size())  # torch.Size([3，30, 63])
    #         print(img_names[0])
    #         # print(imgs[0])
    #         imshow(imgs[0])

    # ======================加载数据方法3
    print('=========方法三==============' * 3)
    '''
      1. 调整数据大小 256*256
      2. 转换pyTorch 张量
      3. 归一化操作
      '''
    simple_transform = transforms.Compose(
        [transforms.Resize((30, 63)), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train = ImageFolder(action_path, simple_transform)
    # 按批加载 pyTorch张量
    train_data_gen = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=1)
    # for imgs, labels in train_data_gen:  # 共（len/32）个batch次循环
    #     print(imgs[0].size())
    #     print(imgs[0])
    #     imshow(imgs[0])
