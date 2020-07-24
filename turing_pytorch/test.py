# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/11/25 19:52
@Describe：

"""
import copy


def test(str_num, case_num):
    case_num += len(str_num)
    if len(str_num) > 2:
        if int(str_num[-2:]) <= 26:
            case_num += 1
        str_num = str_num[:-1]
        test(str_num, case_num)
    else:
        print(case_num)


if __name__ == '__main__':
    a = [1, 2, 3, 4, ['a', 'b']]
    b = a
    c = copy.copy(a)
    print(id(a))
    print(id(b))
    print(id(c))
    print(c)
    print('==========' * 2)
    a.append(5)
    print(a)
    print(c)
    print('==========' * 2)
    a[4].append('c')
    print(a)
    print(c)
    print('==========' * 3)
    a0 = dict(zip(('a', 'b', 'c', 'd', 'e'), (1, 2, 3, 4, 5)))
    print(a0)
    print('==========' * 3)
    fib = lambda n: n if n < 2 else 2 * fib(n - 1)
    print(fib(5))
    print('==========' * 3)
    fib2 = lambda n: n if n < 2 else fib2(n - 1) + fib2(n - 2)
    print(fib2(6))
    print('==========' * 3)
    l = [1, 2, 1, 1, 2]
    print(list(set(l)))
