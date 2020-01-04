#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/4 11:28
# @Author : LYX-夜光

import numpy as np
from os import listdir
from ch02_kNN import kNN

# 读取单个手写文件
def img2vector(fileName):
    returnVect = np.zeros((1, 1024))
    readFile = open(fileName)
    for i in range(32):
        lineStr = readFile.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# 测试分类的正确率
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # traingFileList中每一个项为"A_B.txt"，其中A为手写数字，B为样例序号
        fileNameStr = trainingFileList[i]  # A_B.txt
        fileStr = fileNameStr.split('.')[0]  # A_B
        classNumStr = int(fileStr.split('_')[0])  # A
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]  # A_B.txt
        fileStr = fileNameStr.split('.')[0]  # A_B
        classNumStr = int(fileStr.split('_')[0])  # A
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifierResult = kNN.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %s, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %s" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

if __name__ == "__main__":
    handwritingClassTest()
