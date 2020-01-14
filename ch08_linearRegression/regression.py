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

# 最小二乘公式法求回归参数
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

# 测试局部加权线性回归
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = testArr[i] * lwlr(testArr[i], xArr, yArr, k)
    return yHat

# 标准化
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)
    inVar = np.var(inMat, 0)
    inMat = (inMat - inMeans)/inVar
    return inMat

# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular(奇异), cannot do inverse(转置).")
        return
    weights = denom.I * (xMat.T * yMat)
    return weights

# 数据标准化后测试岭回归
def ridgeTest(xArr, yArr):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        weights = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = weights.T
    return wMat

# 前向逐步线性回归
def stageWise(xArr, yArr, eps=0.01, numIter=100):
    xMat, yMat = np.mat(xArr), np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIter, n))
    weights = np.zeros((n, 1))
    wsMax = weights.copy()
    for i in range(numIter):
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = weights.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if lowestError > rssE:
                    lowestError = rssE
                    wsMax = wsTest
        weights = wsMax.copy()
        returnMat[i, :] = weights.T
    return returnMat

# 计算误差
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
