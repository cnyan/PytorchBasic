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
from common import O_COLUMNS_E, O_COLUMNS, O_COLUMNS_ACC, N_COLUMNS, N_COLUMNS_SPE

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

# 提交第一层数组长度
FIRST_ACTION_DATA_WINDOW_SIZE = 120
# 定义第一层数组窗口的滑动的距离,=1 是默认一帧一帧读取数据
MOVIE_SIZE = 1
# 定义第二层窗口启动长度
SECOND_ACTION_DATA_BEGIN_SIZE = 40
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


# 获得正则化函数
# def get_normalize():
#     """
#     用于dnn_train正则化数据，和测试集的处理,以及基神经网络服务器程序的处理
#     :return:
#     """
#     file_list = glob.glob('D:\\temp\\action_windows\\*.csv')
#
#     action_array = np.zeros(shape=(0, 63))
#     for filename in file_list:
#         df = np.array(pd.read_csv(filename, dtype=float)).round(6)[1:, 1:-1]
#         df = df[:int(len(df) / ACTION_WINDOW_SIZE) * ACTION_WINDOW_SIZE, :]
#
#         # 纵向累加数据
#         action_array = np.concatenate([action_array, df], axis=0)
#
#     # print(action_array.shape)  # (127980, 63)
#     # 对数据进行正则化处理
#     scaler = preprocessing.Normalizer().fit(action_array)
#     # print(scaler)
#     scaler_path = 'src/greater/pre/normalize.pkl'
#     if os.path.exists(scaler_path):
#         os.remove(scaler_path)
#     joblib.dump(scaler, scaler_path)
#     print('正则化建模完毕')


def characteristicFunction(COMPONENTS, read_dir, write_all_feature_file, write_x_de_file):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(read_dir):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            file_list.append(apath)
    print(f'特征提取文件{file_list}')
    all_feature_data = DataFrame(columns=N_COLUMNS_SPE)  # 保存全部（动作）的特征变量
    for file_name in file_list:
        print('{}开始提取特征'.format(file_name))
        # 对应动作类别 1.csv ,2.csv ^ 10.csv
        action_spe = str(file_name.split('\\')[-1]).split('_')[-1].split('.')[0]

        dataMat = DataFrame(pd.read_csv(file_name, names=O_COLUMNS_E)[1:],
                            dtype=float).round(8).drop(['ER', 'ACC'], axis=1)

        # 特征提取
        X = Feature_process(dataMat, ACTION_WINDOW_SIZE, action=str(action_spe))

        # 对特征进行归一化处理
        x_columns = X.columns.values.tolist()
        x_spe = np.array(X.SPE)
        X = preprocessing.normalize(X)
        X = DataFrame(X, columns=x_columns)
        X['SPE'] = x_spe

        # 获取全部的特征数据,然后训练归一化模型函数
        all_feature_data = all_feature_data.append(X, ignore_index=True).round(6)
    # end for
    all_feature_data.to_csv(write_all_feature_file)

    # 特征降维
    X_de = decomposition(all_feature_data, de_str='PCA', n_components=COMPONENTS).round(6)
    X_de.to_csv(write_x_de_file)
    print('特征提取完毕')


def dealwithdynamicdata(dataMat, actionDataWindow):
    # df = pd.read_csv('src/test/origin_data.csv', names=O_COLUMNS)
    count = 1
    # 第一层窗口
    action_data_window_queue = DataFrame(columns=O_COLUMNS_ACC)
    # 第二层窗口
    second_action_data_queue = DataFrame(columns=O_COLUMNS_ACC)
    isActionDetectStart = False  # 动作是否开始
    isActionDetectEnd = False  # 动作是否结束
    actionDetectEndCounter = 0  # 动作结束计数器

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
                isActionDetectEnd = False

            # 如果动作开始，开始提取动作窗口
            if isActionDetectStart:
                second_action_data_queue = second_action_data_queue.append(data, ignore_index=True)

                # 判断动作是否结束
                if (not isActionDetectEnd) and isDnnDetect(action_detect_queue, model='s'):
                    # print('检测到动作结束')
                    isActionDetectEnd = True
                    actionDetectEndCounter = 0  # 启动计数器

                if isActionDetectEnd:
                    actionDetectEndCounter += 1
                    # 此时认为动作结束（结束之后，再添加20个长度）
                    if actionDetectEndCounter > SECOND_ACTION_DATA_BEGIN_SIZE*0.1:
                        isActionDetectStart = False  # 动作提取完毕
                        isActionDetectEnd = False
                        actionDetectEndCounter = 0  # 启动计数器

                        max_index = second_action_data_queue['ACC'].idxmax()
                        # print(str(len(second_action_data_queue)) + ":" + str(max_index))
                        # 动作窗口 action_window
                        action_window = second_action_data_queue[
                                        max_index - int(ACTION_WINDOW_SIZE * 0.5):max_index + int(
                                            ACTION_WINDOW_SIZE * 0.5)]

                        if len(action_window) == ACTION_WINDOW_SIZE:
                            # 提取动作
                            new_action_windows = DataFrame(columns=O_COLUMNS).append(
                                action_window.drop(['ACC'], axis=1),
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


# 快速傅里叶变换
def fft_T_function(dataMat):
    '''
    axis = 0 垂直 做变换
    :param dataMat:
    :return:
    '''
    dataMat = dataMat.T
    dataF = dataMat.apply(np.fft.fft)
    data = dataF.apply(lambda x: np.abs(x.real), axis=1)

    df = []
    for array in data:
        df.append(np.max(array))
    df = np.array(df).reshape(1, 7 * 9)  # 7个节点*9个轴数据
    data = DataFrame(df, columns=O_COLUMNS)
    data = Series(np.array(data)[0], index=O_COLUMNS)
    return data


# end 快速傅里叶变换


# 特征处理
def Feature_process(dataMat, action_data_size, action='1'):
    '''
    将数据集转换为特征矩阵，并标记标签。12一组，60/12=5 ，5个状态
    （一个动作由5维数据表示） 45* 7 = 315 个特征值
    A 均值，C 协方差，K 峰度，S 偏度， F 快速傅里叶（FFT值）
    :param dataMat: 数据集
    :param width: 每个窗口长度（帧数）
    :return: X
    '''
    length = len(dataMat)
    start = 0
    end = action_data_size
    X = DataFrame(columns=N_COLUMNS)

    test_dataFFT = DataFrame()

    while True:
        if start >= length:
            # print('特征处理完毕')
            # 添加特征列
            speceies = []
            for _ in np.arange(len(X['aAXA'])):
                speceies.append(action)
            # 观测傅里叶系数
            # test_dataFFT.to_csv('src/pre/data_fft/dataFFT' + str(action) + '.csv')
            X['SPE'] = speceies
            # print('特征提取完毕...')
            return X
            # break
        else:
            data = dataMat[start:end]

            # 均值 A,协方差C，峰值K,偏度S，
            dataA = data.apply(np.average)
            dataC = data.apply(np.cov)
            # 分别使用df.kurt()方法和df.skew()即可完成峰度he偏度计算
            dataK = data.kurt()
            dataS = data.skew()
            # 使用fft函数对余弦波信号进行傅里叶变换。并取绝对值
            dataF = fft_T_function(data)

            # 保存傅里叶快速变化的值
            test_dataFFT = test_dataFFT.append(data, ignore_index=True)

            df = DataFrame(
                [[dataA.aAX, dataA.aAY, dataA.aAZ, dataC.aAX, dataC.aAY, dataC.aAZ, dataK.aAX, dataK.aAY, dataK.aAZ,
                  dataS.aAX, dataS.aAY, dataS.aAZ, dataF.aAX, dataF.aAY, dataF.aAZ,
                  dataA.aWX, dataA.aWY, dataA.aWZ, dataC.aWX, dataC.aWY, dataC.aWZ, dataK.aWX, dataK.aWY, dataK.aWZ,
                  dataS.aWX, dataS.aWY, dataS.aWZ, dataF.aWX, dataF.aWY, dataF.aWZ,
                  dataA.aHX, dataA.aHY, dataA.aHZ, dataC.aHX, dataC.aHY, dataC.aHZ, dataK.aHX, dataK.aHY, dataK.aHZ,
                  dataS.aHX, dataS.aHY, dataS.aHZ, dataF.aHX, dataF.aHY, dataF.aHZ,

                  dataA.bAX, dataA.bAY, dataA.bAZ, dataC.bAX, dataC.bAY, dataC.bAZ, dataK.bAX, dataK.bAY, dataK.bAZ,
                  dataS.bAX, dataS.bAY, dataS.bAZ, dataF.bAX, dataF.bAY, dataF.bAZ,
                  dataA.bWX, dataA.bWY, dataA.bWZ, dataC.bWX, dataC.bWY, dataC.bWZ, dataK.bWX, dataK.bWY, dataK.bWZ,
                  dataS.bWX, dataS.bWY, dataS.bWZ, dataF.bWX, dataF.bWY, dataF.bWZ,
                  dataA.bHX, dataA.bHY, dataA.bHZ, dataC.bHX, dataC.bHY, dataC.bHZ, dataK.bHX, dataK.bHY, dataK.bHZ,
                  dataS.bHX, dataS.bHY, dataS.bHZ, dataF.bHX, dataF.bHY, dataF.bHZ,

                  dataA.cAX, dataA.cAY, dataA.cAZ, dataC.cAX, dataC.cAY, dataC.cAZ, dataK.cAX, dataK.cAY, dataK.cAZ,
                  dataS.cAX, dataS.cAY, dataS.cAZ, dataF.cAX, dataF.cAY, dataF.cAZ,
                  dataA.cWX, dataA.cWY, dataA.cWZ, dataC.cWX, dataC.cWY, dataC.cWZ, dataK.cWX, dataK.cWY, dataK.cWZ,
                  dataS.cWX, dataS.cWY, dataS.cWZ, dataF.cWX, dataF.cWY, dataF.cWZ,
                  dataA.cHX, dataA.cHY, dataA.cHZ, dataC.cHX, dataC.cHY, dataC.cHZ, dataK.cHX, dataK.cHY, dataK.cHZ,
                  dataS.cHX, dataS.cHY, dataS.cHZ, dataF.cHX, dataF.cHY, dataF.cHZ,

                  dataA.dAX, dataA.dAY, dataA.dAZ, dataC.dAX, dataC.dAY, dataC.dAZ, dataK.dAX, dataK.dAY, dataK.dAZ,
                  dataS.dAX, dataS.dAY, dataS.dAZ, dataF.dAX, dataF.dAY, dataF.dAZ,
                  dataA.dWX, dataA.dWY, dataA.dWZ, dataC.dWX, dataC.dWY, dataC.dWZ, dataK.dWX, dataK.dWY, dataK.dWZ,
                  dataS.dWX, dataS.dWY, dataS.dWZ, dataF.dWX, dataF.dWY, dataF.dWZ,
                  dataA.dHX, dataA.dHY, dataA.dHZ, dataC.dHX, dataC.dHY, dataC.dHZ, dataK.dHX, dataK.dHY, dataK.dHZ,
                  dataS.dHX, dataS.dHY, dataS.dHZ, dataF.dHX, dataF.dHY, dataF.dHZ,

                  dataA.eAX, dataA.eAY, dataA.eAZ, dataC.eAX, dataC.eAY, dataC.eAZ, dataK.eAX, dataK.eAY, dataK.eAZ,
                  dataS.eAX, dataS.eAY, dataS.eAZ, dataF.eAX, dataF.eAY, dataF.eAZ,
                  dataA.eWX, dataA.eWY, dataA.eWZ, dataC.eWX, dataC.eWY, dataC.eWZ, dataK.eWX, dataK.eWY, dataK.eWZ,
                  dataS.eWX, dataS.eWY, dataS.eWZ, dataF.eWX, dataF.eWY, dataF.eWZ,
                  dataA.eHX, dataA.eHY, dataA.eHZ, dataC.eHX, dataC.eHY, dataC.eHZ, dataK.eHX, dataK.eHY, dataK.eHZ,
                  dataS.eHX, dataS.eHY, dataS.eHZ, dataF.eHX, dataF.eHY, dataF.eHZ,

                  dataA.fAX, dataA.fAY, dataA.fAZ, dataC.fAX, dataC.fAY, dataC.fAZ, dataK.fAX, dataK.fAY, dataK.fAZ,
                  dataS.fAX, dataS.fAY, dataS.fAZ, dataF.fAX, dataF.fAY, dataF.fAZ,
                  dataA.fWX, dataA.fWY, dataA.fWZ, dataC.fWX, dataC.fWY, dataC.fWZ, dataK.fWX, dataK.fWY, dataK.fWZ,
                  dataS.fWX, dataS.fWY, dataS.fWZ, dataF.fWX, dataF.fWY, dataF.fWZ,
                  dataA.fHX, dataA.fHY, dataA.fHZ, dataC.fHX, dataC.fHY, dataC.fHZ, dataK.fHX, dataK.fHY, dataK.fHZ,
                  dataS.fHX, dataS.fHY, dataS.fHZ, dataF.fHX, dataF.fHY, dataF.fHZ,

                  dataA.gAX, dataA.gAY, dataA.gAZ, dataC.gAX, dataC.gAY, dataC.gAZ, dataK.gAX, dataK.gAY, dataK.gAZ,
                  dataS.gAX, dataS.gAY, dataS.gAZ, dataF.gAX, dataF.gAY, dataF.gAZ,
                  dataA.gWX, dataA.gWY, dataA.gWZ, dataC.gWX, dataC.gWY, dataC.gWZ, dataK.gWX, dataK.gWY, dataK.gWZ,
                  dataS.gWX, dataS.gWY, dataS.gWZ, dataF.gWX, dataF.gWY, dataF.gWZ,
                  dataA.gHX, dataA.gHY, dataA.gHZ, dataC.gHX, dataC.gHY, dataC.gHZ, dataK.gHX, dataK.gHY, dataK.gHZ,
                  dataS.gHX, dataS.gHY, dataS.gHZ, dataF.gHX, dataF.gHY, dataF.gHZ,
                  ]], columns=N_COLUMNS)

            X = X.append(df)

            start = start + action_data_size
            end = end + action_data_size

    # 特征处理 end


# 数据降维
def decomposition(X, de_str='PCA', n_components=5):
    '''
        对实验数据降维
    :param X:数据集
    :return:X_pca
    '''

    from sklearn.decomposition import PCA
    from sklearn.decomposition import TruncatedSVD

    if de_str == 'PCA':
        de_model = PCA(n_components=n_components)
    else:
        de_model = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)

    X = DataFrame().append(X, ignore_index=True)
    X_data = X.drop('SPE', axis=1)
    # all_decomp_data = DataFrame(columns=['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'SPE'])

    de_model.fit(X_data)
    X_de = de_model.transform(X_data)

    X_de = DataFrame(X_de, columns=['pca' + str(i) for i in np.arange(n_components)])
    X_de['SPE'] = X['SPE']

    # 返回各个成分各自的方差百分比(贡献率) = 0.95
    print('成分各自的方差百分比(贡献率):{}'.format(np.add.reduce(de_model.explained_variance_ratio_)))
    print(de_model.explained_variance_ratio_)

    # columns = ['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'SPE']

    # print('特征降维处理完毕...')
    return X_de


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

    if not is_test_data_process:
        action_root_path = 'D:\\temp\\experiment\\slideWindows\\actionData-6axis'
        extract_action_windows_dir_name = 'D:\\temp\\experiment\\slideWindows\\action_windows-re_dnn-6axis'  # 保存提取后的动作窗口路径
        write_all_feature_file = 'src/greater/pre/all_feature_data_X-6axis.csv'  # 保存特征提取后的全部特征
        write_x_de_file = 'src/greater/pre/action_X_de-6axis.csv'  # 保存特征提取后、特征降维后的全部数据

    else:
        action_root_path = 'D:\\temp\\actionData_test'
        extract_action_windows_dir_name = 'D:/temp/action_windows_test'
        write_all_feature_file = 'src/greater/pre/all_feature_data_X_test.csv'  # 保存特征提取后的全部特征
        write_x_de_file = 'src/greater/pre/action_X_de_test.csv'  # 保存特征提取后、特征降维后的全部数据

    # 提取动作数据 action_window
    readDataAndPropressing(action_root_path, extract_action_windows_dir_name)

    # 提取正则化函数,用于dnn_train正则化数据，和测试集的处理
    # if not is_test_data_process:
    #     get_normalize()

    # 提取特征
    # characteristicFunction(COMPONENTS=10, read_dir=extract_action_windows_dir_name,
    #                       write_all_feature_file=write_all_feature_file, write_x_de_file=write_x_de_file)

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
    """
