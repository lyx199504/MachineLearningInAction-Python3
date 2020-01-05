#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/4 11:17
# @Author : LYX-夜光

import numpy as np

# k近邻算法
def classify0(inX, dataSet, labels, k):
    # inX输入向量 dataSet数据集 labels数据集对应的标签
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # 每行相加
    distances = sqDistances**0.5  # 数据集dataSet和输入向量inX的欧式距离
    sortedDistIndicies = distances.argsort()  # 将距离从小到大排序，排序后数据对应的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 距离最小的k个标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算标签的数量
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

if __name__ == "__main__":
    # 测试
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    print(classify0([0.6, 0.5], group, labels, 3))