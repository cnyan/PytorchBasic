# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/10/28 16:14
@Describe：

"""
import os
import glob
import numpy as np
import platform
import pandas as pd
from PIL import Image
from cnn_base_var import action_window_row, action_window_col

if platform.system() == 'Windows':
    action_root_path = 'D:/temp/action_windows'
    action_image_path = 'D:/home/developer/TrainData/actionImage/allImage'
else:
    action_root_path = '/home/yanjilong/DataSets/action_windows'
    action_image_path = '/home/yanjilong/DataSets/actionImage/allImage'

# 获取文件数组
files_list = glob.glob(os.path.join(action_root_path, '*.csv'))

for file_name in files_list:
    data_mat = np.array(pd.read_csv(file_name), dtype=np.float).round(6)[:, 1:-1]
    data_mat = data_mat[:int(len(data_mat) / action_window_row) * action_window_row, :]  # 确保是30的倍数
    data_mat = np.reshape(data_mat, (-1, action_window_row, action_window_col))

    for i in range(len(data_mat)):
        df = data_mat[i]
        print(df.shape)  # (70, 63)
        df_img = Image.fromarray(np.uint8(df)).convert('RGB')  # 数组转image
        print(np.array(df_img).shape)  # (70, 63,3)
