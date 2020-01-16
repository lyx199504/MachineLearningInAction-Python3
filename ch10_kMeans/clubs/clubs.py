#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/16 20:47
# @Author : LYX-夜光

import numpy as np
from ch10_kMeans import kMeans
import matplotlib.pyplot as plt

# 球面距离
def distSLC(vecA, vecB):
    a = np.sin(vecA[0, 1]*np.pi/180) * np.sin(vecB[0, 1]*np.pi/180)
    b = np.cos(vecA[0, 1]*np.pi/180) * np.cos(vecB[0, 1]*np.pi/180) * np.cos((vecA[0, 0] - vecB[0, 0])*np.pi/180)
    return np.arccos(a + b) * 6371.0

def clusterClubs(numClust=5):
    dataList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        dataList.append([float(lineArr[4]), float(lineArr[3])])  # 经度，纬度
    dataMat = np.mat(dataList)
    centroids, clusterAssment = kMeans.biKmeans(dataMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  # 点的形状
    axprogs = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprogs)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = dataMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle, s=90)
    ax1.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ == "__main__":
    clusterClubs()