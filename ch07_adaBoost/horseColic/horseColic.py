#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/13 10:14
# @Author : LYX-夜光

from ch07_adaBoost import adaBoost
import numpy as np

# 初始化数据
def loadDataSet(fileName):
    lineList = open(fileName).readlines()
    numFeat = len(lineList[0].split('\t'))
    dataMat, labelMat = [], []
    for line in lineList:
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray = adaBoost.adaBoostTrainDS(dataArr, labelArr, 10)
    testData, testLabel = loadDataSet('horseColicTest2.txt')
    prediction10 = adaBoost.adaClassify(testData, classifierArray)
    m = np.shape(testData)[0]
    errArr = np.mat(np.ones((m, 1)))
    errorSum = errArr[prediction10 != np.mat(testLabel).T].sum()
    print("error rate is ", errorSum/m)
