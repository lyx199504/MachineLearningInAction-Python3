#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/15 19:41
# @Author : LYX-夜光

import numpy as np

def loadDataSet(fileName):
    dataMat = []
    readFile = open(fileName)
    for line in readFile.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

# 欧氏距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

# 对每个特征随机生成k个最大与最小之间的数值
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  # 特征值个数
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)  # 随机函数生成k行1列的[0,1)的数组
    return centroids

# K均值算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]  # 数据个数
    clusterAssment = np.mat(np.zeros((m, 2))) - 1  # 第一列为类别序号，第二列为与类中心的误差（距离的平方）
    centroids = createCent(dataSet, k)  # 随机选取k个类中心点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist, minIndex = np.inf, -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 数据与k个类中心的距离
                if minDist > distJI:
                    minDist, minIndex = distJI, j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # 新的类中心
    return centroids, clusterAssment

# 二分K均值
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # 中心点列表
    for j in range(m):  # 所有点距离中心点的距离
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = np.inf
        bestCentToSplit = -1
        bestNewCents, bestClustAss = None, None
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]  # 同一类数据
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 同一类数据再分为两类
            sseSplit = sum(splitClustAss[:, 1])  # 类别i的新距离平方和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 其余类别的旧距离平方和
            if lowestSSE > sseSplit + sseNotSplit:
                lowestSSE = sseSplit + sseNotSplit
                bestCentToSplit = i
                bestNewCents, bestClustAss = centroidMat, splitClustAss.copy()
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 分裂出新类别
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 修改新类中心
        centList.append(bestNewCents[1, :].tolist()[0])  # 新增类中心
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.mat(centList), clusterAssment

# 画出分类图
def plotCluster(dataMat, clusterAssment, k):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(k):
        x = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], 0].T.tolist()[0]
        y = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], 1].T.tolist()[0]
        ax.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    # dataMat = np.mat(loadDataSet('testSet.txt'))
    # k = 4
    # centroids, clusterAssment = kMeans(dataMat, k)
    # plotCluster(dataMat, clusterAssment, k)

    dataMat = np.mat(loadDataSet('testSet2.txt'))
    k = 3
    centList, clusterAssment = biKmeans(dataMat, k)
    plotCluster(dataMat, clusterAssment, k)
