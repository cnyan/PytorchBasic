# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2021/1/8 9:36
@Describe：

"""

"""
可以看到在torch1.6版本中，对torch.save进行了更改
如果在低于1.6版本的服务器中，需要将modle转化格式
torch.save(model.state_dict(), model_cp,_use_new_zipfile_serialization=False)  # 训练所有数据后，保存网络的参数

"""

import os
import glob
import torch
from d_Multi_NN_Net import MyMultiConvNet, MyMultiResCnnNet, MyMultiConvLstmNet, MyMultiConvConfluenceNet, \
    MyMultiTempSpaceConfluenceNet
import platform

if __name__ == '__main__':

    new_model_path = 'src/model'  # 1.6版本以上路径
    no_zip_model_path = 'src/model_no_new_zip'  # 低于1.6版本以下路径
    axis_all = ['6axis', '9axis']

    for axis in axis_all:
        myMultiResCnnNet = MyMultiResCnnNet(int(axis[0]))
        myMultiConvConfluenceNet = MyMultiConvConfluenceNet(int(axis[0]))
        myMultiTempSpaceConfluenceNet = MyMultiTempSpaceConfluenceNet(int(axis[0]))

        models_all = {'myMultiResCnnNet': myMultiResCnnNet,
                      'myMultiConvConfluenceNet': myMultiConvConfluenceNet,
                      'myMultiTempSpaceConfluenceNet':myMultiTempSpaceConfluenceNet}

        for model_name, model_module in models_all.items():
            model_names = os.path.join(new_model_path, f'{model_name}_{axis}_model.pkl')

            model_module.load_state_dict(torch.load(model_names, map_location='cpu'))

            new_model_save_name = os.path.join(no_zip_model_path, f'{model_name}_{axis}_model.pkl')

            torch.save(model_module.state_dict(), new_model_save_name,
                       _use_new_zipfile_serialization=False)  # 训练所有数据后，保存网络的参数
