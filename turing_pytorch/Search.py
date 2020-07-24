# !/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@Author: 闫继龙
@Version: ??
@License: Apache Licence
@CreateTime: 2019/11/26 20:34
@Describe：

"""


def binary_search(list, t):
    """
    从数组中查找元素的位置
    :param list:
    :param t:
    :return:
    """
    low, high = 0, len(list) - 1
    while low < high:
        mid = (low + high) / 2
        if list[mid] > t:
            high = mid
        elif list[mid] < t:
            low = mid
        else:
            return mid
    return low if list[low] == t else False


def qsort(seq):
    """
    快速排序，选中一基准元素，依次将剩余元素中小渝该基准元素的值放置在其左侧，大于该基准元素的值放在右侧；
    然后，对基准元素左右两部分分别进行同样的处理，直至各子序列剩余一个元素时，即排序完成。O(NlogN)
    :param seq:
    :return:
    """
    if seq == []:
        return []
    else:
        pivot = seq[0]
        lesser = qsort([x for x in seq[1:] if x < pivot])
        greater = qsort([x for x in seq[1:] if x > pivot])
        return lesser + [pivot] + greater


def fib(k):
    # fib = lambda n: n if n < 2 else 2 * fib(n - 1)
    # print(fib(5))
    value_fib = lambda n: n if n < 3 else fib(n - 1) + fib(n - 2)
    return value_fib(k)


def coin_change(values, money, coins_used):
    """
    找零问题：
    :param values: T[1:n] 数组
    :param money: 找出来的总钱数
    :param coins_used: 对应于目前钱币总数i所使用的硬币数目
    :return:
    """
    for cents in range(1, money + 1):
        minCoins = cents
        pass


# 二叉树
class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


# 按照层次遍历二叉树(广度遍历)
def lookup(root):
    stack = [root]
    while stack:
        current = stack.pop(0)
        print(current.data, end=' ')
        if current.left:
            stack.append(current.left)
        if current.right:
            stack.append(current.right)
    print()


# 深度遍历
def deep(root):
    if not root:
        return
    print(root.data, end=' ')
    deep(root.left)
    deep(root.right)


# 求最大深度
def max_deep(root):
    if not root:
        return 0
    return max(max_deep(root.left), max_deep(root.right)) + 1


# 判断两棵树是否相同
def is_same_tree(p_root, q_root):
    if p_root == None and q_root == None:
        return True
    elif p_root and q_root:
        return p_root.data == q_root.data and is_same_tree(p_root.left, q_root.left) and is_same_tree(p_root.right,
                                                                                                      q_root.right)
    else:
        return False


# 冒泡排序
def bubble_sort(nums):
    for i in range(len(nums) - 1):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums


if __name__ == '__main__':
    print(fib(5))
    print('=============')
    tree = Node(1, Node(3, Node(7, Node(0)), Node(6)), Node(2, Node(5), Node(4)))
    tree2 = Node(1, Node(4, Node(7, Node(0)), Node(6)), Node(2, Node(5), Node(4)))
    lookup(tree)  # 1 3 2 7 6 5 4 0
    deep(tree)
    print()
    print(max_deep(tree))
    print(is_same_tree(tree, tree2))
