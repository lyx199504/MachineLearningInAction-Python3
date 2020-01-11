#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/10 23:53
# @Author : LYX-夜光

import numpy as np
from ch02_kNN.handWriting.handWriting import img2vector
from ch06_svm.svm import optStruct

# 初始化图像
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        # traingFileList中每一个项为"A_B.txt"，其中A为手写数字，B为样例序号
        fileNameStr = trainingFileList[i]  # A_B.txt
        fileStr = fileNameStr.split('.')[0]  # A_B
        classNumStr = int(fileStr.split('_')[0])  # A
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('../../ch02_kNN/handWriting/trainingDigits')
    opt = optStruct(dataArr, labelArr, 200, 0.0001, kTup)
    opt.smoP(10000)  # 数据训练
    index = np.nonzero(opt.alphas.A > 0)[0]  # 获取大于0的下标
    X, y = opt.X[index], opt.y[index]
    print("there are %d Support Vectors" % np.shape(X)[0])
    errorCount = 0
    for i in range(opt.m):
        kernelEval = opt.kernelTrans(X, opt.X[i, :], kTup)
        predict = kernelEval.T * np.multiply(y, opt.alphas[index]) + opt.b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/opt.m))

    errorCount = 0
    dataArr, labelArr = loadImages('../../ch02_kNN/handWriting/testDigits')
    m = np.shape(dataArr)[0]
    for i in range(m):
        kernelEval = opt.kernelTrans(X, np.mat(dataArr)[i, :], kTup)
        predict = kernelEval.T * np.multiply(y, opt.alphas[index]) + opt.b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

if __name__ == "__main__":
    # 运行需要较长时间
    testDigits(('rbf', 10))