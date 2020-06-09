#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/17 21:54
# @Author : LYX-夜光

import numpy as np
import matplotlib.pyplot as plt

# 初始化文件数据
def loadDataSet(fileName, delim='\t'):
    readFile = open(fileName)
    stringArr = [line.strip().split(delim) for line in readFile.readlines()]
    dataArr = [list(map(float, line)) for line in stringArr]
    return np.mat(dataArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 每个特征数据减去均值
    covMat = np.cov(meanRemoved, rowvar=False)  # 协方差矩阵
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 特征值和特征向量
    eigValIndex = np.argsort(eigVals)  # 下标排序
    eigValIndex = eigValIndex[:-(topNfeat+1):-1]  # 逆序
    redEigVects = eigVects[:, eigValIndex]
    lowDDataMat = meanRemoved * redEigVects  # 降维数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 原始数据的重构
    return lowDDataMat, reconMat

if __name__ == "__main__":
    dataMat = loadDataSet("testSet.txt")
    lowDMat, reconMat = pca(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='o', s=30, c='red')
    plt.show()