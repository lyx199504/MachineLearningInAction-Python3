#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/11 16:04
# @Author : LYX-夜光

from ch02_kNN.handWriting.handWriting import img2vector
import numpy as np
from sklearn.svm import SVC

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
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits():
    trainingMat, hwLabels = loadImages('../../ch02_kNN/handWriting/trainingDigits')
    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)  # 训练

    testMat, hwLabels = loadImages('../../ch02_kNN/handWriting/testDigits')
    errorCount = 0.0
    mTest = testMat.shape[0]
    for i in range(mTest):
        classifierResult = int(clf.predict(np.mat(testMat[i])))
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, hwLabels[i]))
        if classifierResult != hwLabels[i]:
            errorCount += 1.0
    print("\nthe total number of errors is: %s" % errorCount)
    print("the total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    testDigits()
