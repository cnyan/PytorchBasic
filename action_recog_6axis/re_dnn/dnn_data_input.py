# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/10/12 18:39
@Describe：

"""
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import preprocessing
import os
import platform
import joblib
import glob
from dnn_base_var import action_window_col, action_window_row

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if platform.system() == 'Windows':
    # 文件路径
    ACTION_ROOT_PATH = 'D:\\temp\\action_windows-6axis'
    ACTION_ROOT_PATH_TEST = 'D:\\temp\\action_windows_test-6axis'
else:
    ACTION_ROOT_PATH = '/home/yanjilong/DataSets/action_windows-6axis'
    ACTION_ROOT_PATH_TEST = '/home/yanjilong/DataSets/action_windows_test-6axis'


# ER 表示开头序号
def save_data_sets(is_test=False):
    """
    准备数据
    :param is_test: 是否需要分类测试集（单独的数据集合 对应ACTION_ROOT_PATH_TEST）
    :return:
    """
    # 导入数据 8 种动作，每种动作由7个节点数据构成
    file_list = []
    for maindir, subdir, file_name_list in os.walk(ACTION_ROOT_PATH):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            file_list.append(apath)
    # print(file_list)

    # action_array = np.zeros(shape=(0, 1900))
    action_array = np.zeros(shape=(0, (action_window_row * action_window_col) + 1))
    for filename in file_list:
        datamat = []
        labels = []
        print(filename)
        lab = int(filename.split('_')[-1].split('.')[0]) - 1
        label = [lab]
        # label = [0, 0, 0, 0, 0]
        # label[lab] = 1

        df = np.array(pd.read_csv(filename, dtype=float)).round(6)[:, 1:-1]

        df = df[:int(len(df) / action_window_row) * action_window_row, :]

        data = np.reshape(df, (-1, action_window_row, action_window_col))[:900, :]
        # print(data.shape)
        for i in range(len(data)):
            datamat.append(data[i].flatten())
            labels.append(label)
        print(np.array(datamat).shape)  # (行, 40*42)

        # 横向组合数据与标签
        complex_array = np.concatenate([np.array(datamat), np.array(labels)], axis=1)
        # print(complex_array.shape)
        # 纵向累加数据
        action_array = np.concatenate([action_array, complex_array], axis=0)

    print(f'总的数据量为：{action_array.shape}')  # (1299, 2521)   2521=col*row + 1

    # 正则化数据
    scaler = preprocessing.Normalizer().fit(action_array[:, :-1])
    scaler_path = 'src/normalize.pkl'
    joblib.dump(scaler, scaler_path)
    print('正则化建模完毕')

    # 随机排序
    np.random.shuffle(action_array)

    if is_test:
        split_num = -int(len(action_array) * 0.1)
        action_array = action_array[:split_num]
        test_action_data = action_array[split_num:]

    if os.path.exists('src/train_action_data.npy'):
        os.remove('src/train_action_data.npy')
    np.save('src/train_action_data.npy', action_array)

    if is_test:
        print(f'总的测试集数据：{len(test_action_data)}行')
        if os.path.exists('src/test_action_data.npy'):
            os.remove('src/test_action_data.npy')
        np.save('src/test_action_data.npy', test_action_data)


# 不做正则化处理，在预测的时候，正则化
def save_data_test():
    # 导入数据 8 种动作，每种动作由7个节点数据构成
    file_list = glob.glob(os.path.join(ACTION_ROOT_PATH_TEST, '*'))
    # action_array = np.zeros(shape=(0, 1900))
    action_array = np.zeros(shape=(0, (action_window_row * action_window_col) + 1))
    for filename in file_list:
        datamat = []
        labels = []

        lab = int(filename.split('_')[-1].split('.')[0]) - 1
        label = [lab]
        # label = [0, 0, 0, 0, 0]
        # label[lab] = 1

        df = np.array(pd.read_csv(filename, dtype=float)).round(6)[:, 1:-1]
        df = df[:int(len(df) / action_window_row) * action_window_row, :]

        data = np.reshape(df, (-1, action_window_row, action_window_col))
        # print(data.shape)
        for i in range(len(data)):
            datamat.append(data[i].flatten())
            labels.append(label)
        print(np.array(datamat).shape)
        # print(np.array(labels).shape)
        # 横向组合数据与标签
        complex_array = np.concatenate([np.array(datamat), np.array(labels)], axis=1)
        # print(complex_array.shape)
        # 纵向累加数据
        action_array = np.concatenate([action_array, complex_array], axis=0)
        # action_dict = {'data': np.array(datamat), 'labels': np.array(labels)}
        # print(action_dict['data'])
        # print('======' * 5)

    print(f'测试集数据量为：{action_array.shape}')  # (11077, 1900)  1890+10

    df_data = action_array[:, :-1]
    df_labels = np.array(action_array[:, -1:])

    # （不做正则化测试集） 测试集预测的时候，做处理
    # normalize = joblib.load('src/normalize.pkl')
    # df_data = normalize.transform(df_data)

    action_array = np.concatenate([df_data, df_labels], axis=1)
    # 随机排序
    np.random.shuffle(action_array)

    if os.path.exists('src/test_action_data.npy'):
        os.remove('src/test_action_data.npy')

    np.save('src/test_action_data.npy', action_array)


def read_data_sets(file_path='src/train_action_data.npy', valid_size=0.2):
    '''
    :param file_path: 需要读取的文件路径
    :param test_size: 随机数，用来区分训练集和测试机
    :return:
    '''

    df_array = np.load(file_path)
    df_data = df_array[:, :-1]
    df_labels = df_array[:, -1:]

    # 正则化数据处理
    normalize = joblib.load('src/normalize.pkl')
    df_data = normalize.transform(df_data)

    # df_data = preprocessing.normalize(df_data)

    df_array = np.concatenate([df_data, df_labels], axis=1)
    # print(df_array[0, :10])
    # 重新排序
    np.random.shuffle(df_array)

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(df_array[:, :-1], df_array[:, -1:], test_size=valid_size)

    print('总的训练数据集:' + str(len(X_train)) + '行')
    print('总的验证数据集:' + str(len(X_valid)) + '行')

    action_sets = {'all_data': df_array[:, :-1], 'all_labels': df_array[:, -1:], 'train_data': X_train,
                   'train_label': y_train, 'validation_data': X_valid,
                   'validation_label': y_valid, 'original_data': df_array}
    return action_sets


if __name__ == '__main__':
    # 保存数据
    save_data_sets(is_test=True)

    # 保存测试集数据(使用第三人的测试数据集)
    # save_data_test()
    # 数据分类
    actions_data = read_data_sets(file_path='src/train_action_data.npy', valid_size=0.2)
