#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/3 13:40
# @Author : LYX-夜光

import numpy as np

# k近邻算法
def classify0(inX, dataSet, labels, k):
    # inX输入向量 dataSet数据集 labels数据集对应的标签
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  # 每行相加
    distances = sqDistances**0.5  # 数据集dataSet和输入向量inX的欧式距离
    sortedDistIndicies = distances.argsort()  # 将距离从小到大排序，排序后数据对应的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 距离最小的k个标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 计算标签的数量
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

# 初始化数据
def file2matrix(fileName):
    readFile = open(fileName)
    arrayOfLines = readFile.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 将每个样本数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每列最小值
    maxVals = dataSet.max(0)  # 每列最大值
    ranges = maxVals - minVals
    m = dataSet.shape[0]  # 数据行数
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 测试分类的正确率
def datingClassTest():
    """
        datingTestSet2.txt中数据：
        第一列：每年获得的飞行常客里程数
        第二列：玩视频游戏所耗时间百分比
        第三列：每周消费的冰淇淋公升数
        第四列：三种类型的人：1不喜欢的人 2魅力一般的人 3极具魅力的人
        """
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):  # 前numTestVecs个数据为测试集，其他数据为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# 判断某人的分类
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person: %s" % resultList[classifierResult-1])

if __name__ == "__main__":
    datingClassTest()
    classifyPerson()

    # # 绘制散点图
    # import matplotlib.pyplot as plt
    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    # ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15*np.array(datingLabels), 15*np.array(datingLabels))
    # plt.show()
