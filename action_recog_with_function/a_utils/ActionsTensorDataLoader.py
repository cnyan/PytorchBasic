# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/23 18:33
@Describe：

"""
from glob import glob
import numpy as np
import platform
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler


class ActionsTensorDataLoader():
    def __init__(self, img_data_root, size=(7 * 9, 36)):
        """
        :param img_data_path:
        :param size:  width = 63,height = 30
        """
        super().__init__()
        self.files = glob(img_data_root + '*.jpg')
        self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
