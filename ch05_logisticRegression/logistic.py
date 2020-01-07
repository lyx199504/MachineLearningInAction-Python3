#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/7 11:12
# @Author : LYX-夜光

import numpy as np
import random

# sigmoid函数
def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

# 梯度下降
def gradDescent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        weights -= alpha * dataMatrix.transpose() * (h - labelMat)  # 梯度下降法
    return weights

# alpha动态减少机制下的随机梯度下降
def stocGradDescent(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            weights -= alpha * (h - classLabels[randIndex]) * dataMatrix[randIndex]
            del dataIndex[randIndex]
    return weights

# 画出拟合曲线
def plotBestFit(dataMat, labelMat, weights):
    import matplotlib.pyplot as plt
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]  # z=w0+w1*x+w2*y
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == "__main__":
    dataMat, labelMat = [], []
    readFile = open('testSet.txt')
    for line in readFile.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    weights = stocGradDescent(np.array(dataMat), labelMat, 500)
    plotBestFit(dataMat, labelMat, weights)
