# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/12/5 16:01
@Describe：

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


# e^x = 1 + x + x^2/2!+...
def calc_e_small(x):
    n = 10
    # 累乘  cumsum是求和
    # 1! 2! 3! 4! 5!...10!
    f = np.arange(1, n + 1).cumprod()
    # x x^2 ... x^10
    b = np.array([x] * n).cumprod()
    return 1 + np.sum(b / f)


'''
e^x = ln2 + (e^ln2)/1!*(x-ln2) + (e^ln2)/2!*(x-ln2)^2+...
x = a*ln2 + b   k<= z  |b| <= 1/2ln2
a = ln( int( x/ln2 + 0.5 ) )
b = x-a*ln2
e^x = 2^a + e^b
'''


def calc_e(x):
    reverse = False
    if x < 0:  # 处理负数  exp(-x) = 1/exp(x)
        x = -x
        reverse = True
    ln2 = 0.69314718055994530941723212145818
    c = x / ln2
    a = int(c + 0.5)
    b = x - a * ln2
    # 2的a次方乘以e的b次幂
    y = (2 ** a) * calc_e_small(b)
    if reverse:
        return 1 / y
    return y


if __name__ == '__main__':
    # -2到0 十个数
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    # 0到2 二十个数
    t2 = np.linspace(0, 2, 20)
    t = np.concatenate((t1, t2))
    print(t)  # 横轴数据
    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, '=', y[i], '(近似值)\t', math.exp(x))
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', linewidth=2)
    plt.plot(t, y, 'go', linewidth=2)
    plt.title(u'Taylor展开式的应用', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True)
    plt.show()
