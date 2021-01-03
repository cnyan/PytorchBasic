# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2020/12/28 10:13
@Describe：

"""
import os
import copy
import numpy as np
import time
import platform
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import AUtils
import datetime

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
    def __init__(self, modelNet, model_name, cls, mean_std):
        super(NN_Predict, self).__init__()
        self.model = modelNet
        self.cls = cls

        self.model.load_state_dict(torch.load(f'src/model/{model_name}_{cls}_model.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        self.model_name = model_name

        mean = mean_std[0]
        std = mean_std[1]

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        if platform.system() == 'Windows':
            test_dir = fr'D:/home/DataRec/Action_Test/actionImage/{cls}/test'
        else:
            test_dir = fr'/home/yanjilong/dataSets/DataRec/Action_Test/actionImage/{cls}/test'

        self.test_action_data_set = ImageFolder(test_dir, transform=data_transforms)
        print(f'test_data size:{len(self.test_action_data_set)}')

    def predict(self):
        rights = []
        labels = []
        for data, label in self.test_action_data_set:
            data = data.unsqueeze(0)  # 扩展一个维度
            label = torch.LongTensor([int(label)])
            if torch.cuda.is_available():
                data = data.cuda()

            labels.append(label)
            output = self.model(data)

            right = self.rightness(output, label)
            rights.append(right)

        # 计算校验集的平均准确度
        right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
        print("模式{}-{},准确率：{:.3f},识别个数：{}".format(model_name,cls,right_ratio, len(labels)))
        AUtils.metrics(np.array(labels), np.array([i[3] for i in rights]).flatten())
        AUtils.plot_confusion_matrix(np.array(labels), np.array([i[3] for i in rights]).flatten(),
                                     classes=[0, 1, 2, 3, 4],
                                     savePath=f'src/test_plt_img/{self.model_name}_{self.cls}_predict.png',
                                     title=f'{self.model_name}_{self.cls}_predict')

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

    """
    窗口长度 36
    xyz 6 (14, 36, 3)
    xyz 9 (21, 36, 3)
    awh 9 (21, 36, 3)
    org 6 (42, 36, 3)
    org 9 (63, 36, 3)
    """
    from AUtils import make_print_to_file
    from Single_NN_Net import MyConvNet, MyDnn, MyDilConvNet

    make_print_to_file()

    acls = ['xyz-6axis', 'xyz-9axis', 'org-6axis', 'org-9axis', 'awh-9axis']
    acls_scale = [(3, 14, 36), (3, 21, 36), (3, 42, 36), (3, 63, 36), (3, 21, 36)]
    mean_stds = [([0.3, 0.47, 0.46], [0.26, 0.35, 0.32]), ([0.34, 0.49, 0.47], [0.26, 0.32, 0.31]),
                 ([0.4, 0.4, 0.4], [0.42, 0.42, 0.42]), ([0.43, 0.43, 0.43], [0.39, 0.39, 0.39]),
                 ([0.33, 0.49, 0.49], [0.31, 0.32, 0.27])]

    for i, cls in enumerate(acls):
        scale = acls_scale[i]
        myDnn = MyDnn(scale[0] * scale[1] * scale[2])
        myCnn = MyConvNet(scale[0], (scale[1], scale[2]))
        myDilCnn = MyDilConvNet(scale[0], (scale[1], scale[2]))

        models = {'MyDnn': myDnn, 'MyCnn': myCnn, 'MyDilCnn': myDilCnn}
        # models = {'MyCnn': myCnn}

        for model_name, model in models.items():
            print('===================******** begin  test test test  begin *********=================')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            nn_predict = NN_Predict(model, model_name, cls, mean_stds[i])
            with Timer() as t:
                nn_predict.predict()
            print('predict time {0}'.format(str(t.interval)[:5]))

            print('===================******** end  test test test  end *********=================')
