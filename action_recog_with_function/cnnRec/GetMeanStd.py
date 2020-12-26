# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/25 11:27
@Describe：

"""

import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Get_Mean_Std():
    def __init__(self, opt, cls):
        # 训练，验证，测试数据集文件夹名
        self.opt = opt
        self.dirs = ['train', 'test', 'test']
        self.cls = cls

        self.means = [0, 0, 0]
        self.stdevs = [0, 0, 0]

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 数据值从[0,255]范围转为[0,1]，相当于除以255操作
            # transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])

        # 因为这里使用的是ImageFolder，按文件夹给数据分类，一个文件夹为一类，label会自动标注好
        self.dataset = {x: ImageFolder(os.path.join(opt, x), self.transform) for x in self.dirs}

    def get_mean_std(self, type):
        """
        计算数据集的均值和标准差
        :param type: 使用的是那个数据集的数据，有'train', 'test', 'testing'
        :param mean_std_path: 计算出来的均值和标准差存储的文件
        :return:
        """
        num_imgs = len(self.dataset[type])
        for data in self.dataset[type]:
            img = data[0]
            for i in range(3):
                # 一个通道的均值和标准差
                self.means[i] += img[i, :, :].mean()
                self.stdevs[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs

        print("{} : normMean = {}".format(self.cls, np.round(self.means, 2)))
        print("{} : normstdevs = {}".format(self.cls, np.round(self.stdevs, 2)))


if __name__ == '__main__':
    root = r'D:\home\DataRec\actionImage'
    acls = ['xyz-6axis', 'xyz-9axis', 'org-6axis', 'org-9axis', 'awh-9axis']
    for cls in acls:
        getmeanstd = Get_Mean_Std(os.path.join(root, cls), cls)
        getmeanstd.get_mean_std('train')
