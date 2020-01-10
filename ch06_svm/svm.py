#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/8 15:59
# @Author : LYX-夜光

import numpy as np
import random

# 初始化数据
def loadDataSet(fileName):
    dataMat, labelMat = [], []
    readFile = open(fileName)
    for line in readFile.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# 从0-m内随机选择一个不同于i的下标
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

# 调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    alphas = np.mat(np.zeros((m, 1)))
    b = 0
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # KTT条件公式为：f = (W^T)x + b = alpha*label*x*(x[i]^T) + b
            fXi = float(np.multiply(alphas, labelMat).T * (dataMat*dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])  # 使f趋向于label，即使(W^T)x + b - label ≈ 0，E为误差
            if labelMat[i] * Ei < -toler and alphas[i] < C or labelMat[i] * Ei > toler and alphas[i] > 0:
                # labelMat[i] * Ei为：label*[(W^T)x + b] - label^2 = label*[(W^T)x + b] - 1 ≈ 0
                # toler为误差的容忍度，可视为0附近的正数；C为最大间隔，小于1.0
                # labelMat[i] * Ei < -toler 意味着f较小，需增大alpha，且alpha不能大于C
                # labelMat[i] * Ei > toler 意味着f较大，需减小alpha，且alpha不能小于0
                j = selectJrand(i, m)  # 随机选取下标j
                fXj = float(np.multiply(alphas, labelMat).T * (dataMat*dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - dataMat[i, :] * dataMat[i, :].T - dataMat[j, :] * dataMat[j, :].T
                if eta >= 0:  # 公式：eta = 2XiXj - Xi^2 - Xj^2，我也不懂
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta  # 调整alpha，公式我也不懂
                alphas[j] = clipAlpha(alphas[j], H, L)  # 将alpha限制在闭区间[L, H]中
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 下面这一坨我也不知道是怎么来的，很乱，等我理解透彻再更新注释吧！！！
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMat[i, :]*dataMat[i, :].T \
                     - labelMat[j]*(alphas[j]-alphaJold)*dataMat[i, :]*dataMat[j, :].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMat[i, :]*dataMat[j, :].T \
                     - labelMat[j]*(alphas[j]-alphaJold)*dataMat[j, :]*dataMat[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

# 画出拟合曲线
def plotFit(dataMat, labelMat, alphas, b):
    import matplotlib.pyplot as plt
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 0])
            ycord1.append(dataArr[i, 1])
        else:
            xcord2.append(dataArr[i, 0])
            ycord2.append(dataArr[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-1.0, 8.0, 0.1)
    weights = np.multiply(alphas, np.mat(labelMat).transpose()).T * np.mat(dataMat)
    weights = np.array(weights)[0]
    y = (-np.array(b)[0][0]-weights[0]*x)/weights[1]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("testSet.txt")
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(dataMat[i], labelMat[i])
    plotFit(dataMat, labelMat, alphas, b)