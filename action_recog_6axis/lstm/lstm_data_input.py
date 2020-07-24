# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/12/21 10:26
@Describe：

"""
from glob import glob
import os
import shutil
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from PIL import Image
import platform

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cnn_base_var import action_window_row, action_window_col


def save_data_img_with_color(data_mat, action_image_dir_path):
    action_image_dir_num = action_image_dir_path[-1]
    print(data_mat.shape)
    for i in range(len(data_mat)):
        df = data_mat[i]
        df_max = data_mat[i].max()
        df = df * 255. / df_max  # 使所有数据小于255
        # df = df.astype(int)

        img_name = action_image_dir_num + '_' + str(i) + '.jpg'
        plt.imsave(os.path.join(action_image_dir_path, img_name), df)
        # action_img = Image.fromarray(df,mode='P') # RGB三通道黑白，L单通道彩色
        # action_img.save(os.path.join(action_image_dir_path, img_name))


def save_data_img_with_origin(data_mat, action_image_dir_path):
    action_image_dir_num = action_image_dir_path[-1]
    print(data_mat.shape)
    for i in range(len(data_mat)):
        df = data_mat[i]

        # action_img 替换一下三行代码
        df_max = data_mat[i].max()
        df = df * 255. / df_max
        # df = df.astype(int)

        action_img = Image.fromarray(df.astype('uint8')).convert('RGB')  # RGB三通道黑白，L单通道彩色
        img_name = action_image_dir_num + '_' + str(i) + '.jpg'
        action_img.save(os.path.join(action_image_dir_path, img_name))

        # if i == 0:
        #     print(np.array(action_img).transpose([2, 0, 1]))
        #     plt.imshow(np.array(action_img))
        #     plt.show()


def save_data_img_with_reshape(data_mat, action_image_dir_path):
    zero_array = np.zeros(46)
    action_image_dir_num = action_image_dir_path.split('/')[-1]
    for i in range(len(data_mat)):
        df = data_mat[i]

        df_max = data_mat[i].max()
        df = df * 255. / df_max
        df = data_mat[i].flatten()
        df = np.append(df, zero_array).reshape(44, 44)
        df = df.astype(int)

        img_name = action_image_dir_num + '_' + str(i) + '.jpg'
        action_img = Image.fromarray(df, mode='I').convert('RGB')  # RGB三通道黑白，L单通道彩色
        action_img.save(os.path.join(action_image_dir_path, img_name))
        # action_img.show()
        # print(df)
        # plt.imshow(df)
        # plt.show()


def create_all_image_from_csv(action_root_path, action_image_path, save_model='origin'):
    # 创建工作路径
    if os.path.exists(action_image_path[:-9]):
        for folder in os.listdir(action_image_path[:-9]):
            folder_path = os.path.join(action_image_path[:-9], folder)
            shutil.rmtree(folder_path)
        os.mkdir(action_image_path)

    # 获取文件数组
    files_list = glob(os.path.join(action_root_path, '*.csv'))

    for file_name in files_list:
        data_mat = np.array(pd.read_csv(file_name), dtype=np.float).round(6)[:, 1:-1]
        data_mat = data_mat[:int(len(data_mat) / action_window_row) * action_window_row, :]  # 确保是30的倍数
        data_mat = np.reshape(data_mat, (-1, action_window_row, action_window_col))

        # 创建保存动作图片的文件夹
        action_image_dir_num = file_name.split('/')[-1].split('_')[-1].split('.')[0]
        action_image_dir_path = os.path.join(action_image_path, action_image_dir_num)

        if not os.path.exists(action_image_dir_path):
            os.mkdir(action_image_dir_path)

        print(file_name)

        if save_model == 'origin':
            # 保存图片,实现方式一
            save_data_img_with_origin(data_mat, action_image_dir_path)
        elif save_model == 'reshape':
            # 实现方式二
            save_data_img_with_reshape(data_mat, action_image_dir_path)
        else:
            # 实现方式三，保存最初的数据
            save_data_img_with_color(data_mat, action_image_dir_path)

    # img = Image.open(action_image_path + '/1/1_0.jpg')
    # img = Image.open('D:\\home\\developer\\TrainData\\dogsandcats\\test1\\1.jpg')
    # img = np.array(img)
    # print('======' * 6)
    # print(img)
    # plt.imshow(img)
    # plt.show()
    # print('图片形状' + str(img.shape))
    # print(img)


# 构造训练集和验证机
def create_train_valid(action_image_path, valid_size=0.1, test_size=0.1):
    files_list = glob(os.path.join(action_image_path, '*'))
    img_path = action_image_path[:-9]

    # 创建目录
    for t in ['train', 'valid', 'test']:
        if not os.path.exists(os.path.join(img_path, t)):
            os.mkdir(os.path.join(img_path, t))
        if t == 'test':
            continue
        for folder in [i[-1] for i in files_list]:
            folder_path = os.path.join(img_path, t, folder)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
    print('文件目录创建完毕')

    # 将图片的一小部分子集复制到valid文件夹中
    for action_class in files_list:
        actions_num = glob(os.path.join(action_class, "*.jpg"))

        shuffle = np.random.permutation(actions_num)  # 乱序文件索引
        # print(shuffle)

        valid_len = int(len(shuffle) * valid_size)
        for i in shuffle[:valid_len]:
            if platform.system() == 'Windows':
                os.rename(i, os.path.join(img_path, 'valid', action_class[-1], i.split('\\')[-1]))
            else:
                os.rename(i, os.path.join(img_path, 'valid', action_class[-1], i.split('/')[-1]))
    print(f"valid创建完成,valid数目是：{len(glob(os.path.join(img_path, 'valid', '*/*.jpg')))}")

    # 将图片的一小部分子集复制到test文件夹中
    for action_class in files_list:
        actions_num = glob(os.path.join(action_class, "*.jpg"))
        shuffle = np.random.permutation(actions_num)  # 乱序文件索引
        test_len = int(len(shuffle) * test_size)
        for i in shuffle[:test_len]:
            if platform.system() == 'Windows':
                os.rename(i, os.path.join(img_path, 'test', i.split('\\')[-1]))
            else:
                os.rename(i, os.path.join(img_path, 'test', i.split('/')[-1]))
    print(f"test创建完成,test数目是：{len(glob(os.path.join(img_path, 'test', '*.jpg')))}")

    # 将另一部图片复制到train文件夹中
    for action_class in files_list:
        actions_num = glob(os.path.join(action_class, "*.jpg"))
        for i in actions_num:
            if platform.system() == 'Windows':
                os.rename(i, os.path.join(img_path, 'train', action_class[-1], i.split('\\')[-1]))
            else:
                os.rename(i, os.path.join(img_path, 'train', action_class[-1], i.split('/')[-1]))
    print(f"train创建完成,train数目是：{len(glob(os.path.join(img_path, 'train', '*/*.jpg')))}")


if __name__ == '__main__':
    if platform.system() == 'Windows':
        action_root_path = 'D:/temp/action_windows'
        action_image_path = 'D:/home/developer/TrainData/actionImage/allImage'
    else:
        action_root_path = '/home/yanjilong/DataSets/action_windows'
        action_image_path = '/home/yanjilong/DataSets/actionImage/allImage'

    # save_model='color' || 'origin' || 'reshape'
    create_all_image_from_csv(action_root_path, action_image_path, save_model='origin')
    create_train_valid(action_image_path, valid_size=0.2, test_size=0.2)
