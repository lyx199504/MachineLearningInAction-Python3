#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/18 13:54
# @Author : LYX-夜光

from ch13_pca import pca
import numpy as np

# 用列均值代替NaN值
def replaceNanWithMean():
    dataMat = pca.loadDataSet('secom.data', ' ')
    numFeat = np.shape(dataMat)[1]  # 特征数
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
        dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat

if __name__ == "__main__":
    dataMat = replaceNanWithMean()
    lowDDataMat, reconMat = pca.pca(dataMat, 6)
    print(lowDDataMat)