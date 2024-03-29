# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/28 10:02
@Describe：

"""
import platform
from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

np.set_printoptions(suppress=True)


class DataToImg():
    def __init__(self, windowDataFoldPath, imgDataFoldPath, axis='9axis'):
        self.windowDataFoldPath = windowDataFoldPath
        self.imgDataFoldPath = imgDataFoldPath
        self.internalNodeNum = 7
        self.action_window_row = 36  # 窗口长度
        self.axis = axis
        self.action_window_col = self.internalNodeNum * int(axis[1])

    def showImg(self, data):
        # plt.imshow(data)
        # plt.show()
        # 可视化图像
        plt.figure(figsize=(3, 3))
        plt.imshow(data, cmap=plt.cm.gray)
        plt.axis('on')  # 不显示坐标
        plt.show()
        plt.close()

    def changeTo_org_img(self, dataMat, img_name):
        """
        原始图片数据L灰度
        :param dataMat:
        :return:
        """
        # df_max = dataMat.max()
        # dataMat = dataMat * 255. / df_max

        dataMat = np.uint8(dataMat).T
        img_data = Image.fromarray(dataMat).convert('L')

        if img_name[-6:-4] == '_0':
            self.showImg(img_data)
            print(np.array(img_data).shape)
        img_data.save(img_name)

    def changeTo_awh_img(self, dataMat, img_name):
        """
        按照 加速度，角速度，磁场 重构3维数据
        :param dataMat:
        :return:
        """
        dataMat = np.uint8(dataMat)

        A1 = [x * 9 for x in range(0, 7)]
        A2 = [x + 1 for x in A1]
        A3 = [x + 2 for x in A1]
        A = np.append(A1, A2)
        A = np.append(A, A3)
        A.sort()  # 7个节点的 加速度集合，7*3
        W = [x + 3 for x in A]  # 7个节点的 角速度集合，7*3
        H = [x + 3 for x in W]  # 7个节点的 磁场集合，7*3

        RA = dataMat[:, A].T
        GW = dataMat[:, W].T
        BH = dataMat[:, H].T

        r = Image.fromarray(RA, mode='L')
        g = Image.fromarray(GW, mode='L')
        b = Image.fromarray(BH, mode='L')

        pic = Image.merge('RGB', (r, g, b))  # 合并三通道
        # r, g, b = pic.split()  # 分离三通道
        if img_name[-6:-4] == '_0':
            self.showImg(pic)
            print(np.array(pic).shape)
        pic.save(img_name)

    def changeTo_xyz_img(self, dataMat, img_name):
        """
        按照 X,Y,Z轴 重构3维数据
        :param dataMat:
        :return:
        """
        # df_max = dataMat.max()
        # dataMat = dataMat * 255. / df_max

        dataMat = np.uint8(dataMat)

        X = [x * 3 for x in range(0, (int(self.axis[1]) // 3) * 7)]
        Y = [x * 3 + 1 for x in range(0, (int(self.axis[1]) // 3) * 7)]
        Z = [x * 3 + 2 for x in range(0, (int(self.axis[1]) // 3) * 7)]

        RX = dataMat[:, X].T  # AWH的X轴集合 3*7
        GY = dataMat[:, Y].T  # AWH的Y轴集合 3*7
        BZ = dataMat[:, Z].T  # AWH的Z轴集合 3*7

        r = Image.fromarray(RX, mode='L')
        g = Image.fromarray(GY, mode='L')
        b = Image.fromarray(BZ, mode='L')

        pic = Image.merge('RGB', (r, g, b))  # 合并三通道
        # r, g, b = pic.split()  # 分离三通道
        if img_name[-6:-4] == '_0':
            self.showImg(pic)
            print(np.array(pic).shape)
        pic.save(img_name)

    def readWindowsToImageData(self, model='xyz'):
        """
        读取窗口数据文件，并转为图像（x,y,z作为三个通道）
        :param model: xyz, awh
        :return:
        """
        files_list = glob(os.path.join(self.windowDataFoldPath, '*.csv'))

        for file_name in files_list:
            print(file_name)
            # 创建保存动作图片的文件夹
            image_dir_num = int(file_name[-5]) - 1
            img_save_path = os.path.join(self.imgDataFoldPath, str(image_dir_num))  # 存储路径
            if not os.path.exists(img_save_path):
                os.mkdir(img_save_path)

            data_mat = pd.read_csv(file_name, dtype=float, header=0).round(3)
            data_mat = np.array(data_mat)[:, 1:-1]
            # if self.axis == '-6axis':
            #     drop_col = ['aHX', 'aHY', 'aHZ', 'bHX', 'bHY', 'bHZ', 'cHX', 'cHY', 'cHZ', 'dHX', 'dHY', 'dHZ', 'eHX',
            #                 'eHY', 'eHZ', 'fHX', 'fHY', 'fHZ', 'gHX', 'gHY', 'gHZ', ]
            #     data_mat = np.array(data_mat.drop(drop_col, axis=1))[:, 1:-1]  # 删除第一列序号列和ACC列
            # else:
            #     data_mat = np.array(data_mat)[:, 1:-1]

            data_mat = data_mat[:int(len(data_mat) / self.action_window_row) * self.action_window_row, :]  # 确保是30的倍数
            data_mat = np.reshape(data_mat, (-1, int(self.action_window_row), int(self.action_window_col)))
            # print(data_mat.shape)
            data_mat = data_mat[:180, :, ]  # 每个动作取1600个 (1600, 40, 63)

            # print(data_mat.shape)

            for i, df in enumerate(data_mat):
                img_name = str(image_dir_num) + '_' + str(i) + '.jpg'
                img_name = os.path.join(img_save_path, img_name)

                if model == 'xyz':
                    self.changeTo_xyz_img(df, img_name)
                elif model == 'awh':
                    self.changeTo_awh_img(df, img_name)
                elif model == 'org':
                    self.changeTo_org_img(df, img_name)
                else:
                    raise

    # 构造训练集和验证机
    def create_test_folder(self, action_image_path):
        """
        创建测试仪集合
        :param action_image_path:
        :param valid_size:
        :param test_size:
        :return:
        """
        files_list = glob(os.path.join(action_image_path, '*'))
        img_path = action_image_path[:-9]

        # 创建目录
        for t in ['train', 'valid', 'test']:
            mkdir_path = os.path.join(img_path, t)
            if os.path.exists(mkdir_path):
                shutil.rmtree(mkdir_path)

            if t == 'test':
                os.makedirs(os.path.join(img_path, t))
            else:
                continue

            # 创建 1，2，3，4，5文件夹
            for folder in [i[-1] for i in files_list]:
                folder_path = os.path.join(img_path, t, folder)
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
        print('文件目录创建完毕')

        # 将图片的一小部分子集复制到test文件夹中
        for action_class in files_list:
            actions_num = glob(os.path.join(action_class, "*.jpg"))
            actions_num.sort(key=lambda x: int(x.split('\\')[-1].split('_')[-1].split('.')[0]))
            # shuffle = np.random.permutation(actions_num)  # 乱序文件索引
            test_len = int(len(actions_num))

            for i in actions_num[0:test_len]:
                if platform.system() == 'Windows':
                    os.rename(i, os.path.join(img_path, 'test', action_class[-1], i.split('\\')[-1]))
                else:
                    os.rename(i, os.path.join(img_path, 'test', action_class[-1], i.split('/')[-1]))
        print(f"test创建完成,test数目是：{len(glob(os.path.join(img_path, 'test', '*/*.jpg')))}")



if __name__ == '__main__':
    axiss = ['-6axis', '-9axis']  # 9轴和6轴
    models = ['xyz', 'org', 'awh']  # 三种模式 xyz awh org
    """
    窗口长度 36
    xyz 6 (14, 36, 3)
    xyz 9 (21, 36, 3)
    awh 9 (21, 36, 3)
    org 6 (42, 36, 1)
    org 9 (63, 36, 1)
    """
    for model in models:
        for axis in axiss:

            if model == 'awh' and axis == '-6axis':
                continue

            if platform.system() == 'Windows':
                action_root_path = f'D:/home/DataRec/Action_Test/action_windows{axis}'
                action_image_path = f'D:/home/DataRec/Action_Test/actionImage/{model}{axis}/allImage'
                if os.path.exists(action_image_path):
                    shutil.rmtree(action_image_path)
            else:
                action_root_path = f'/home/yanjilong/dataSets/DataRec/Action_Test/action_windows{axis}'
                action_image_path = f'/home/yanjilong/dataSets/DataRec/Action_Test/actionImage/{model}{axis}/allImage'
                if os.path.exists(action_image_path):
                    shutil.rmtree(action_image_path)

            if not os.path.exists(action_image_path):
                os.makedirs(action_image_path)

            dataToImgCls = DataToImg(action_root_path, action_image_path, axis=axis)
            dataToImgCls.readWindowsToImageData(model=model)
            print(f'当前状态：modl={model},axis={axis}')
            dataToImgCls.create_test_folder(action_image_path)
