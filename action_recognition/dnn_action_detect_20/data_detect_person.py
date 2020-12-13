#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import os
import math
import time
from tqdm import tqdm, trange
import glob
import joblib
from sklearn import preprocessing, linear_model
import matplotlib as mpl
from matplotlib import pyplot as plt
import warnings
import torch.nn.functional as F
from common import O_COLUMNS_E, O_COLUMNS, O_COLUMNS_ACC, N_COLUMNS, N_COLUMNS_SPE

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

# 提交第一层数组长度
FIRST_ACTION_DATA_WINDOW_SIZE = 120
# 定义第一层数组窗口的滑动的距离,=1 是默认一帧一帧读取数据
MOVIE_SIZE = 1
# 定义第二层窗口启动长度
SECOND_ACTION_DATA_BEGIN_SIZE = 30
# 第三层数组长度（需要识别的动作数组,提取到的动作窗口）
ACTION_WINDOW_SIZE = 40
# 定义探测数组的长度（是否开始运动）
ACTION_DETECT_ACTION_BEGIN_SIZE = -20
# 定义探测数组的斜率阈值
# ACTION_DETECT_K_START_VALUE = 15
# ACTION_DETECT_K_END_VALUE = -7

# ACTION_ROOT_PATH = '../dataPropress/data/'

# 计数器：记录动作开始的个数
upCount = 0

import torch
from torch import nn
from torch.autograd import Variable

class dnn_action_detect(nn.Module):
    def __init__(self):
        super(dnn_action_detect, self).__init__()
        self.layer1 = torch.nn.Linear(20, 400)
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(400, 400)
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.Linear(400, 2)
        self.layer6 = torch.nn.LogSoftmax(dim=1)

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class Predict():
    def __init__(self):
        super().__init__()
        self.mode_path = 'src/dnn_action_model.pkl'
        self.scaler = joblib.load('src/normalize.pkl')

        self.model = dnn_action_detect()
        # # 加载模型(GPU或CPU模型)
        model_weights = torch.load(self.mode_path, map_location=lambda storage, loc: storage)

        self.model.load_state_dict(model_weights.state_dict())

        # self.model = torch.load(self.mode_path)
        self.model.eval()  # 指定测试

    def predict(self, array):
        x = self.scaler.transform(np.array([array])).flatten()  # 正则化数据
        x = torch.FloatTensor(x).view(1, -1)
        y_predict = self.model(x).detach().numpy()

        predict = np.array([math.pow(math.e, i) for i in y_predict.flatten()])

        y_max = predict.max()
        y_argmax = predict.argmax()

        return y_max, y_argmax

        # 基于DNN的深度探测网络


dnn_detect = Predict()


# ER 表示开头序号
def readDataAndPropressing(action_root_path, write_dir_name, SHOW_PLOT=False):
    '''
    :param beginActionModel: 动作开始的判断方法，regression 或 slope
    :param SHOW_PLOT:
    :return:
    '''
    # 导入数据 8 种动作，每种动作由7个节点数据构成
    file_list = []
    for maindir, subdir, file_name_list in os.walk(action_root_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            file_list.append(apath)
    print(f'提取动作文件{file_list}')

    # file_list = ['D:\\temp\\actionData-6axis\\5.csv']
    all_feature_data = DataFrame(columns=N_COLUMNS_SPE)  # 保存全部（动作）的特征变量
    for file_name in file_list:
        # 对应动作的文件后缀名 1.csv ,2.csv ^ 10.csv
        toCsvFileSuffix = str(file_name.split('\\')[-1])

        print('第{0}个动作,探测阈值是{1}'.format(toCsvFileSuffix.split('.')[0], 'null'))

        dataMat = DataFrame(pd.read_csv(file_name, names=O_COLUMNS_E), dtype=float).round(8).drop(['ER', 'ACC'],
                                                                                                  axis=1)
        print(dataMat.shape)
        actionDataWindow = DataFrame(columns=O_COLUMNS)

        actionDataWindow = dealwithdynamicdata(dataMat, actionDataWindow)

        print(actionDataWindow.shape)
        print('提取动作{0}完毕'.format(toCsvFileSuffix.split('.')[0]))

        if len(actionDataWindow) < ACTION_WINDOW_SIZE: continue
        # 计算右手手腕10号传感器（对应2组数据）在每个采样时刻 t 所对应的三轴加速度合成加速度，确定击球点
        # actionDataWindow['ACC'] = actionDataWindow.apply(lambda row:
        #                                                  (float(row['bAX']) + float(row['bAY']) + float(row['bAZ'])),
        #                                                  axis=1)
        actionDataWindow['ACC'] = actionDataWindow.apply(
            lambda row: math.sqrt(float(row['bAX']) ** 2 + float(row['bAY']) ** 2 + float(row['bAZ'] ** 2)), axis=1)

        if not os.path.exists(write_dir_name):
            os.mkdir(write_dir_name)
        # 保存提取后的数据
        actionDataWindow.to_csv(os.path.join(write_dir_name, 'action_data_' + toCsvFileSuffix))
    # end for
    print('动作窗口提取完毕')



def dealwithdynamicdata(dataMat, actionDataWindow):
    # df = pd.read_csv('src/test/origin_data.csv', names=O_COLUMNS)
    count = 1
    # 第一层窗口
    action_data_window_queue = DataFrame(columns=O_COLUMNS_ACC)
    # 第二层窗口
    second_action_data_queue = DataFrame(columns=O_COLUMNS_ACC)
    isActionDetectStart = False  # 动作是否开始

    actionCount = 0
    with tqdm(total=len(dataMat)) as pbar:  # 设置进度条
        # begin while
        while count < len(dataMat):

            pbar.update(1)  # 更新进度条

            if len(action_data_window_queue) == FIRST_ACTION_DATA_WINDOW_SIZE:
                action_data_window_queue = action_data_window_queue[MOVIE_SIZE:]
            # 模拟数据一帧 一帧的读取
            data = dataMat.loc[count]

            # data['ACC'] = float(data['bAX']) + float(data['bAY']) + float(data['bAZ'])
            data['ACC'] = math.sqrt((data['bAX']) ** 2 + float(data['bAY']) ** 2 + float(data['bAZ']) ** 2)

            # action_data_window_queue = action_data_window_queue.append(data,ignore_index=True)
            action_data_window_queue = action_data_window_queue.append(data)
            # print(action_data_window_queue)
            # 探测动作斜率(开始或结束)
            action_detect_queue = action_data_window_queue['ACC'][ACTION_DETECT_ACTION_BEGIN_SIZE:].values

            # 判断动作是否开始
            if (not isActionDetectStart) and isDnnDetect(action_detect_queue, model='a'):
                # print('检测到动作开始')
                isActionDetectStart = True
                second_action_data_queue = action_data_window_queue[-SECOND_ACTION_DATA_BEGIN_SIZE:]

            # 重新开始
            if len(second_action_data_queue) > FIRST_ACTION_DATA_WINDOW_SIZE:
                isActionDetectStart = False

            # 如果动作开始，开始提取动作窗口
            if isActionDetectStart:
                second_action_data_queue = second_action_data_queue.append(data, ignore_index=True)
                # 此时认为动作结束
                if len(second_action_data_queue) > ACTION_WINDOW_SIZE * 1.8:
                    isActionDetectStart = False  # 动作提取完毕

                    max_index = second_action_data_queue['ACC'].idxmax()
                    # print(str(len(second_action_data_queue)) + ":" + str(max_index))
                    # 动作窗口 action_window
                    action_window = second_action_data_queue[
                                    max_index - int(ACTION_WINDOW_SIZE * 0.5):max_index + int(ACTION_WINDOW_SIZE * 0.5)]

                    if len(action_window) == ACTION_WINDOW_SIZE:
                        # 提取动作
                        new_action_windows = DataFrame(columns=O_COLUMNS).append(action_window.drop(['ACC'], axis=1),
                                                                                 ignore_index=True)
                        # print(f'new action window {len(new_action_windows)}')
                        actionDataWindow = actionDataWindow.append(new_action_windows, ignore_index=True)

                        actionCount += 1
                    else:
                        pass
                    # print('提取窗口长度 %s' % len(action_window))
            # end if(count 循环计数器）
            count += 1
        # end while

    global upCount
    print('检测到动作开始次数 %s' % upCount)
    print('提取出动作:{}'.format(actionCount))
    upCount = 0
    return actionDataWindow


def isDnnDetect(array, model='a'):
    if len(array) < 20:
        return None
    y_max, y_argmax = dnn_detect.predict(np.array(array))

    if y_max > 0.9:
        if model == 'a':  # 加速判断
            if y_argmax == 0:
                global upCount
                upCount += 1
                return True  # 上升
            else:
                return False  # 下降
        elif model == 's':  # 减速判断
            if y_argmax == 1:
                return True  # 下降
            else:
                return False  # 上升
        else:
            return False




class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


if __name__ == '__main__':
    # 设定是否设置为测试模式
    is_test_data_process = False

    """
    # # 全部数据集
    # if not is_test_data_process:
    #     action_root_path = 'D:\\temp\\experiment\\person\\everyPerson\\actionData-6axis\\P6'
    #     extract_action_windows_dir_name = 'D:\\temp\\experiment\\person\\everyPerson\\action_windows'  # 保存提取后的动作窗口路径
    #     write_all_feature_file = 'src/greater/pre/all_feature_data_X-6axis.csv'  # 保存特征提取后的全部特征
    #     write_x_de_file = 'src/greater/pre/action_X_de-6axis.csv'  # 保存特征提取后、特征降维后的全部数据
    # 
    # else:
    #     action_root_path = 'D:\\temp\\actionData_test'
    #     extract_action_windows_dir_name = 'D:/temp/action_windows_test'
    #     write_all_feature_file = 'src/greater/pre/all_feature_data_X_test.csv'  # 保存特征提取后的全部特征
    #     write_x_de_file = 'src/greater/pre/action_X_de_test.csv'  # 保存特征提取后、特征降维后的全部数据
    """

    if not is_test_data_process:
        action_root_path = 'D:\\temp\\actionData-6axis'
        extract_action_windows_dir_name = 'D:/temp/action_windows-6axis'  # 保存提取后的动作窗口路径
        write_all_feature_file = 'src/greater/pre/all_feature_data_X-6axis.csv'  # 保存特征提取后的全部特征
        write_x_de_file = 'src/greater/pre/action_X_de-6axis.csv'  # 保存特征提取后、特征降维后的全部数据

    else:
        action_root_path = 'D:\\temp\\actionData_test'
        extract_action_windows_dir_name = 'D:/temp/action_windows_test'
        write_all_feature_file = 'src/greater/pre/all_feature_data_X_test.csv'  # 保存特征提取后的全部特征
        write_x_de_file = 'src/greater/pre/action_X_de_test.csv'  # 保存特征提取后、特征降维后的全部数据


    # 提取动作数据 action_window
    readDataAndPropressing(action_root_path, extract_action_windows_dir_name)


