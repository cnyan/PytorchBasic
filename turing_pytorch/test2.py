# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/11/29 13:11
@Describe：

"""
import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


with Timer() as t:
    i = 1 + 1
    time.sleep(1)
    i = i + 2

print("took %.03f sec" % t.interval)
