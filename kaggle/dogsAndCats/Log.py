# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/9/2 14:45
@Describe：

"""

import sys
import os
import platform
import datetime
import time


class Logger(object):
    def __init__(self, filename=None):
        today = datetime.date.today()
        if filename == None:
            filename = today.__str__() + '.txt'

        rootDir = ''
        if platform.system() == 'Windows':
            rootDir = 'D:\home\developer\LogData'
        else:
            rootDir = '/home/yanjilong/LogData'
        if not os.path.exists(rootDir):
            os.mkdir(rootDir)

        self.terminal = sys.stdout
        self.filePath = os.path.join(rootDir, filename)
        self.log = open(self.filePath, "a")

    def write(self, msg):
        if msg == '\n':
            return
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f'{time_str}----{msg} \n'
        self.terminal.write(message)
        # self.log.write(message)
        with open(self.filePath, 'a+') as f:
            f.write(message)

    def flush(self):
        pass


''' 
sys.stdout = Logger()
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()

print(path)
print(os.path.dirname(__file__))
print('***' * 5)
'''