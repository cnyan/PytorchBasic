# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/30 20:25
@Describe：

"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataTestToTorch import ActionTestDataSets
import AUtils
import time
import warnings

warnings.filterwarnings('ignore')


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


class NN_Predict():
    def __init__(self, modelNet, model_name, axis):
        super(NN_Predict, self).__init__()
        self.model = modelNet
        self.axis = axis

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{axis}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.model_name = model_name

        action_data_test_set = ActionTestDataSets(axis=axis)
        self.test_action_data_set = DataLoader(action_data_test_set, shuffle=True, num_workers=2)
        print(f'test_data shape: ({len(action_data_test_set)}{(action_data_test_set.data_shape())})')

    def predict(self):
        rights = []
        labels = []
        for data, label in self.test_action_data_set:
            # data = data.unsqueeze(0)  # 扩展一个维度
            label = torch.LongTensor([int(label)])
            if torch.cuda.is_available():
                data = data.cuda()

            labels.append(label)
            output = self.model(data)

            right = self.rightness(output, label)
            rights.append(right)

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("模式{}-{},准确率：{:.3f},识别个数：{}".format(self.model_name, self.axis, right_ratio, len(labels)))

        AUtils.metrics(np.array(labels), np.array([i[3] for i in rights]).flatten())
        AUtils.plot_confusion_matrix(np.array(labels), np.array([i[3] for i in rights]).flatten(),
                                     classes=[0, 1, 2, 3, 4],
                                     savePath=f'src/test_plt_img/{self.model_name}_{self.axis}_predict.png',
                                     title=f'{self.model_name}_{self.axis}_predict')

    # 自定义计算准确度的函数
    def rightness(self, predict, label):
        '''
        计算准确度
        :param predict:
        :param label:
        :return: right,len(label),score,pred_idx
        '''
        prob = F.softmax(predict, dim=1)
        score = torch.max(prob, 1)[0].cpu().data.numpy()[0]
        pred_idx = torch.max(predict, 1)[1].cpu()  # 最大值下标(类别)
        right = pred_idx.eq(label.data.view_as(pred_idx)).sum()  # 返回 0（false)，1(true)
        return right.data.item(), len(label), score, pred_idx.cpu().data.numpy()[0]


if __name__ == '__main__':
    from AUtils import make_print_to_file  # 打印日志
    from d_Multi_NN_Net import MyMultiConvNet, MyMultiResCnnNet, MyMultiConvLstmNet, MyMultiConvConfluenceNet

    make_print_to_file()
    if torch.cuda.is_available():
        torch.cuda.set_device(1)

    axis_all = ['9axis', '6axis']

    for axis in axis_all:
        myMultiConvNet = MyMultiConvNet(int(axis[0]))
        myMultiResCnnNet = MyMultiResCnnNet(int(axis[0]))
        myMultiConvLstmNet = MyMultiConvLstmNet(int(axis[0]))
        myMultiConvConfluenceNet = MyMultiConvConfluenceNet(int(axis[0]))

        models_all = {'myMultiConvNet': myMultiConvNet, 'myMultiResCnnNet': myMultiResCnnNet,
                      'myMultiConvLstmNet': myMultiConvLstmNet,'myMultiConvConfluenceNet': myMultiConvConfluenceNet}

        models_all = {'myMultiResCnnNet': myMultiResCnnNet,'myMultiConvConfluenceNet': myMultiConvConfluenceNet}

        for model_name, model in models_all.items():
            print('===================********begin begin begin*********=================')
            print(f'当前执行参数：model={model_name}_{axis}')

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            nn_predict = NN_Predict(model, model_name, axis=axis)
            with Timer() as t:
                nn_predict.predict()
            print('predict time {0}'.format(str(t.interval)[:5]))
