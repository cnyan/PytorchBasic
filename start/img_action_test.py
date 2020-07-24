# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/9/26 9:10
@Describe：

"""
from glob import glob
import os
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import matplotlib.pyplot as plt

action_root_path = 'D:\\temp\\action_windows'
action_image_path = 'D:\\home\\developer\\TrainData\\actionImage\\allImage'

# 获取文件数组
files_list = glob(os.path.join(action_root_path, '*.csv'))
zero_array = np.zeros(46)
for file_name in files_list:
    data_mat = np.array(pd.read_csv(file_name), dtype=np.float).round(6)[:, 1:-1]
    data_mat = data_mat[:int(len(data_mat) / 30) * 30, :]  # 确保是30的倍数
    data_mat = np.reshape(data_mat, (-1, 30, 63))

    # 创建保存动作图片的文件夹
    action_image_dir_num = file_name.split('\\')[-1].split('_')[-1].split('.')[0]
    action_image_dir_path = os.path.join(action_image_path, action_image_dir_num)
    if not os.path.exists(action_image_dir_path):
        os.mkdir(action_image_dir_path)

    break
    print(file_name)

    for i in range(len(data_mat)):
        df = data_mat[i]
        df_max = data_mat[i].max()
        df = df * 255. / df_max
        df = data_mat[i].flatten()
        df = np.append(df, zero_array).reshape(44, 44)
        df = df.astype(int)

        img_name = action_image_dir_num + '_' + str(i) + '.jpg'
        action_img = Image.fromarray(df).convert('RGB') # RGB三通道黑白，L单通道彩色
        action_img.save(os.path.join(action_image_dir_path, img_name))
        # action_img.show()
        # print(df)
        # plt.imshow(df)
        # plt.show()
        # break

img = Image.open('D:\\home\\developer\\TrainData\\actionImage\\allImage\\1\\1_0.jpg')
# img = Image.open('D:\\home\\developer\\TrainData\\dogsandcats\\test1\\1.jpg')
img = np.array(img)
plt.imshow(img)
plt.show()
print(img.shape)
print(img)
