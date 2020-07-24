#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: 'AD'
@license: Apache Licence 
@time: 2018/10/11 11:21
Describe：
    
    
"""
# ER 表示开头序号
O_COLUMNS_E = ['ER', 'aAX', 'aAY', 'aAZ', 'aWX', 'aWY', 'aWZ', 'bAX', 'bAY', 'bAZ', 'bWX', 'bWY',
               'bWZ', 'cAX', 'cAY', 'cAZ', 'cWX', 'cWY', 'cWZ', 'dAX', 'dAY', 'dAZ', 'dWX', 'dWY', 'dWZ',
               'eAX', 'eAY', 'eAZ', 'eWX', 'eWY', 'eWZ', 'fAX', 'fAY', 'fAZ', 'fWX', 'fWY', 'fWZ',
               'gAX', 'gAY', 'gAZ', 'gWX', 'gWY', 'gWZ', 'ACC']

O_COLUMNS = O_COLUMNS_E[1:-1]

O_COLUMNS_ACC = O_COLUMNS_E[1:]

# （1号）加速度的时频特征+ 角速度时频特征 + 磁场的时频特征 + 2号+……
N_COLUMNS_SPE = ['aAXA', 'aAYA', 'aAZA', 'aAXC', 'aAYC', 'aAZC', 'aAXK', 'aAYK', 'aAZK', 'aAXS', 'aAYS', 'aAZS', 'aAXF',
                 'aAYF', 'aAZF',
                 'aWXA', 'aWYA', 'aWZA', 'aWXC', 'aWYC', 'aWZC', 'aWXK', 'aWYK', 'aWZK', 'aWXS', 'aWYS', 'aWZS', 'aWXF',
                 'aWYF', 'aWZF',
                 'aHXA', 'aHYA', 'aHZA', 'aHXC', 'aHYC', 'aHZC', 'aHXK', 'aHYK', 'aHZK', 'aHXS', 'aHYS', 'aHZS', 'aHXF',
                 'aHYF', 'aHZF',
                 'bAXA', 'bAYA', 'bAZA', 'bAXC', 'bAYC', 'bAZC', 'bAXK', 'bAYK', 'bAZK', 'bAXS', 'bAYS', 'bAZS', 'bAXF',
                 'bAYF', 'bAZF',
                 'bWXA', 'bWYA', 'bWZA', 'bWXC', 'bWYC', 'bWZC', 'bWXK', 'bWYK', 'bWZK', 'bWXS', 'bWYS', 'bWZS', 'bWXF',
                 'bWYF', 'bWZF',
                 'bHXA', 'bHYA', 'bHZA', 'bHXC', 'bHYC', 'bHZC', 'bHXK', 'bHYK', 'bHZK', 'bHXS', 'bHYS', 'bHZS', 'bHXF',
                 'bHYF', 'bHZF',
                 'cAXA', 'cAYA', 'cAZA', 'cAXC', 'cAYC', 'cAZC', 'cAXK', 'cAYK', 'cAZK', 'cAXS', 'cAYS', 'cAZS', 'cAXF',
                 'cAYF', 'cAZF',
                 'cWXA', 'cWYA', 'cWZA', 'cWXC', 'cWYC', 'cWZC', 'cWXK', 'cWYK', 'cWZK', 'cWXS', 'cWYS', 'cWZS', 'cWXF',
                 'cWYF', 'cWZF',
                 'cHXA', 'cHYA', 'cHZA', 'cHXC', 'cHYC', 'cHZC', 'cHXK', 'cHYK', 'cHZK', 'cHXS', 'cHYS', 'cHZS', 'cHXF',
                 'cHYF', 'cHZF',
                 'dAXA', 'dAYA', 'dAZA', 'dAXC', 'dAYC', 'dAZC', 'dAXK', 'dAYK', 'dAZK', 'dAXS', 'dAYS', 'dAZS', 'dAXF',
                 'dAYF', 'dAZF',
                 'dWXA', 'dWYA', 'dWZA', 'dWXC', 'dWYC', 'dWZC', 'dWXK', 'dWYK', 'dWZK', 'dWXS', 'dWYS', 'dWZS', 'dWXF',
                 'dWYF', 'dWZF',
                 'dHXA', 'dHYA', 'dHZA', 'dHXC', 'dHYC', 'dHZC', 'dHXK', 'dHYK', 'dHZK', 'dHXS', 'dHYS', 'dHZS', 'dHXF',
                 'dHYF', 'dHZF',
                 'eAXA', 'eAYA', 'eAZA', 'eAXC', 'eAYC', 'eAZC', 'eAXK', 'eAYK', 'eAZK', 'eAXS', 'eAYS', 'eAZS', 'eAXF',
                 'eAYF', 'eAZF',
                 'eWXA', 'eWYA', 'eWZA', 'eWXC', 'eWYC', 'eWZC', 'eWXK', 'eWYK', 'eWZK', 'eWXS', 'eWYS', 'eWZS', 'eWXF',
                 'eWYF', 'eWZF',
                 'eHXA', 'eHYA', 'eHZA', 'eHXC', 'eHYC', 'eHZC', 'eHXK', 'eHYK', 'eHZK', 'eHXS', 'eHYS', 'eHZS', 'eHXF',
                 'eHYF', 'eHZF',
                 'fAXA', 'fAYA', 'fAZA', 'fAXC', 'fAYC', 'fAZC', 'fAXK', 'fAYK', 'fAZK', 'fAXS', 'fAYS', 'fAZS', 'fAXF',
                 'fAYF', 'fAZF',
                 'fWXA', 'fWYA', 'fWZA', 'fWXC', 'fWYC', 'fWZC', 'fWXK', 'fWYK', 'fWZK', 'fWXS', 'fWYS', 'fWZS', 'fWXF',
                 'fWYF', 'fWZF',
                 'fHXA', 'fHYA', 'fHZA', 'fHXC', 'fHYC', 'fHZC', 'fHXK', 'fHYK', 'fHZK', 'fHXS', 'fHYS', 'fHZS', 'fHXF',
                 'fHYF', 'fHZF',
                 'gAXA', 'gAYA', 'gAZA', 'gAXC', 'gAYC', 'gAZC', 'gAXK', 'gAYK', 'gAZK', 'gAXS', 'gAYS', 'gAZS', 'gAXF',
                 'gAYF', 'gAZF',
                 'gWXA', 'gWYA', 'gWZA', 'gWXC', 'gWYC', 'gWZC', 'gWXK', 'gWYK', 'gWZK', 'gWXS', 'gWYS', 'gWZS', 'gWXF',
                 'gWYF', 'gWZF',
                 'gHXA', 'gHYA', 'gHZA', 'gHXC', 'gHYC', 'gHZC', 'gHXK', 'gHYK', 'gHZK', 'gHXS', 'gHYS', 'gHZS', 'gHXF',
                 'gHYF', 'gHZF', 'SPE']

N_COLUMNS = N_COLUMNS_SPE[:-1]
