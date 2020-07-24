# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/8/27 17:36
@Describe：

"""
import glob
import os
import numpy as np
import platform

'''
python在模块glob中定义了glob()函数，实现了对目录内容进行匹配的功能，
glob.glob()函数接受通配模式作为输入，并返回所有匹配的文件名和路径名列表，
与os.listdir类似。

函数np.random.shuffle与np.random.permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）;
区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。

'''

if __name__ == '__main__':
    path = ''

    if platform.system() == 'Windows':
        path = 'D:/home/developer/TrainData/dogsandcats'
    else:
        path = '/home/yanjilong/DataSets/dogsandcats'

    files = glob.glob(os.path.join(path, 'train/*.jpg'))
    print(f'total no of images {len(files)}')

    no_of_images = len(files)
    # 创建可用于验证数据集的混合索引
    shuffle = np.random.permutation(no_of_images)
    try:
        # 创建保存验证图片集的validation目录
        os.mkdir(os.path.join(path, 'valid'))
    except FileExistsError as e:
        print('valid folder already exist')

    # 使用标签名称创建目录
    for t in ['train', 'valid']:
        for folder in ['dog/', 'cat/']:
            if os.path.exists(os.path.join(path, t, folder)):
                continue
            os.mkdir(os.path.join(path, t, folder))

    if len(glob.glob(os.path.join(path, 'valid', '*/*.jpg'))) != 2000:  # 只复制一次

        # 将图片的一小部分子集复制到validation文件夹
        for i in shuffle[:2000]:
            # print(files[i])  # D:/home/developer/TrainData/dogsandcats\train\dog.7055.jpg
            folder = files[i].split('\\')[-1].split('.')[0]  # cat dot
            image = files[i].split('\\')[-1]  # 图片名称
            os.rename(files[i], os.path.join(path, 'valid', folder, image))
        print('验证集数据复制完毕')

        # 将图片的一小部分子集复制到training文件夹中
        for i in shuffle[2000:]:
            folder = files[i].split('\\')[-1].split('.')[0]  # cat dot
            image = files[i].split('\\')[-1]  # 图片名称
            os.rename(files[i], os.path.join(path, 'train', folder, image))
        print('训练集数据复制完毕')
    else:
        print('数据准备完毕...')
