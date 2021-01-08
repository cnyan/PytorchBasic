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
from d_Multi_NN_Net import MyMultiResCnnNet, MyMultiConvConfluenceNet
import platform

load_model_name = 'myMultiResCnnNet'  # @@@@@@@@@@ 修改1
new_model_path = 'src/model'  # 1.6版本以上路径
no_zip_model_path = 'src/model_no_new_zip'  # 低于1.6版本以下路径

if __name__ == '__main__':
    model_names = glob.glob(os.path.join(new_model_path, f'{load_model_name}*.pkl'))
    model_names.sort(key=lambda x: int(x[-15]))

    for model_name in model_names:
        axis  =  int(model_name[-15])

        model = MyMultiResCnnNet(int(axis))    # @@@@@@@@@@ 修改2
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        if platform.system() == 'Windows':
            model_name = model_name.split("\\")[-1]
        else:
            model_name = model_name.split(r'/')[-1]

        torch.save(model.state_dict(), os.path.join(no_zip_model_path, model_name.split(r'/')[-1]),
                   _use_new_zipfile_serialization=False)  # 训练所有数据后，保存网络的参数
