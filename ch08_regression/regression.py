#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/13 23:10
# @Author : LYX-夜光

import numpy as np

# 初始化数据
def loadDataSet(fileName):
    lineList = open(fileName).readlines()
    numFeat = len(lineList[0].split('\t')) - 1
    dataMat, labelMat = [], []
    for line in lineList:
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

# 公式法求回归参数
def standRegress(xArr, yArr):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 行列式
        print("This matrix is singular(奇异), cannot do inverse(转置).")
        return
    weights = xTx.I * (xMat.T * yMat)  # [(X^T)X]^-1 * (X^T) * y
    return weights

# 局部加权线性回归LWLR
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    m = np.shape(xMat)[0]
    W = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        W[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (W * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular(奇异), cannot do inverse(转置).")
        return
    weights = xTx.I * (xMat.T * (W * yMat))
    return weights

# 计算预测值
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = testArr[i] * lwlr(testArr[i], xArr, yArr, k)
    return yHat

# 最小二乘
def rssError(yMat, yHat):
    return ((yMat-yHat)**2).sum()

# 画出拟合曲线
def plotBestFit(dataMat, labelMat, yHat):
    import matplotlib.pyplot as plt
    xArr = np.array(dataMat)
    yArr = np.array(labelMat)
    yHat = np.array(yHat)
    indexSort = xArr[:, 1].argsort(0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xArr[indexSort, 1], yArr[indexSort], s=30, c='red', marker='s')
    ax.plot(xArr[indexSort, 1], yHat[indexSort])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    # xArr, yArr = loadDataSet('ex0.txt')
    # weights = standRegress(xArr, yArr)
    # yMat = np.mat(yArr)
    # yHat = np.mat(xArr)*weights
    # print("线性回归：预测值与真实值的相关度为", np.corrcoef(yHat.T, yMat)[0, 1])
    # plotBestFit(xArr, yArr, yHat)

    # 局部加权线性回归
    xArr, yArr = loadDataSet('ex0.txt')
    testX, testY = loadDataSet('ex1.txt')
    k = 0.01  # 参数
    yHat = lwlrTest(testX, xArr, yArr, k)
    plotBestFit(testX, testY, yHat)
