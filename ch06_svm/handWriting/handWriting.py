#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/10 23:53
# @Author : LYX-夜光

import numpy as np
from ch02_kNN.handWriting.handWriting import img2vector

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
    dataArr, labelArr = loadImages('trainingDigits')
    # b, alphas =

if __name__ == "__main__":
    print(loadImages("../../ch02_kNN/handWriting/trainingDigits"))