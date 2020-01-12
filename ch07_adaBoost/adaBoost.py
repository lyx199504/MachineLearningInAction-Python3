#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/11 19:52
# @Author : LYX-夜光

import numpy as np

# 按单个特征分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':  # less than
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

# 根据单个特征分类，选取最好的分类结果
def buildStump(dataArr, classLabels, D):
    dataMat, labelMat = np.mat(dataArr), np.mat(classLabels).T
    m, n = np.shape(dataMat)
    numSteps = 10
    bestStump, minError, bestClass = {}, np.inf, np.mat(np.zeros((m, 1)))
    for i in range(n):
        rangeMin, rangeMax = dataMat[:, i].min(), dataMat[:, i].max()  # 每个特征的最小和最大值
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j)*stepSize
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # 预测值与实际值相等为0，不等为1
                weightedError = D.T * errArr
                if minError > weightedError:
                    minError = weightedError
                    bestClass = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClass

# adaBoost算法，训练分类
def adaBoostTrainDS(dataArr, classLabels, numIter=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)  # 初始化1/m的权重
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(numIter):
        bestStump, minError, bestClass = buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        # 计算alpha公式：alpha = 1/2*ln((1-epsilon)/epsilon)
        alpha = float(0.5*np.log((1.0-minError)/max(minError, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("bestClass: ", bestClass.T)
        # 调整权重D公式：
        #   样本分类正确：D(i+1) = D(i)*exp(-alpha)/sum(D(i))
        #   样本分类错误：D(i+1) = D(i)*exp(alpha)/sum(D(i))
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, bestClass)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClass += alpha*bestClass  # sign(aggClass)为分类结果
        print("aggClass: ", aggClass.T)
        # 计算分类错误率
        aggErrors = np.multiply(np.sign(aggClass) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr

# 预测分类
def adaClassify(dataArr, classifierArr):
    dataMat = np.mat(dataArr)
    m = np.shape(dataMat)[0]
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        bestClass = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClass += classifierArr[i]['alpha'] * bestClass
    return np.sign(aggClass)

if __name__ == "__main__":
    dataMat = np.mat([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    classifierArr = adaBoostTrainDS(dataMat, classLabels, 9)
    testData = [[0, 0], [2, 2]]
    results = adaClassify(testData, classifierArr)
    for i in range(len(testData)):
        print(testData[i], "classified as: ", int(results[i]))