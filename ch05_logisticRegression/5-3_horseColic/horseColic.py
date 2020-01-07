#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/7 22:03
# @Author : LYX-夜光

from ch05_logisticRegression import logistic
import numpy as np

# 逻辑回归分类
def classifyVector(inX, weights):
    prob = logistic.sigmoid(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

# 测试疝气病马数据
def colicTest():
    readTrain = open('horseColicTraining.txt')
    readTest = open('horseColicTest.txt')
    trainingSet, trainingLabels = [], []
    for line in readTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logistic.stocGradDescent(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in readTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

# 多次平均错误率
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/numTests))

if __name__ == "__main__":
    multiTest()