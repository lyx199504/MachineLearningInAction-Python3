#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/4 13:18
# @Author : LYX-夜光

from math import log
import json

# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 标签出现的概率
        shannonEnt -= prob * log(prob, 2)  # 香农熵公式
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    returnDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            returnDataSet.append(reducedFeatVec)
    return returnDataSet

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        uniqueVals = set([example[i] for example in dataSet])
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 信息增益
        if bestInfoGain < infoGain:  # 取信息增益最大
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 选出数量最多的类别
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[-1], reverse=True)
    return sortedClassCount[0][0]

# 构建决策树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del labels[bestFeat]
    uniqueVals = set([example[bestFeat] for example in dataSet])
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 使用决策树分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict:
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':  # 子树为非叶节点
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:  # 子树为叶子节点
                classLabel = secondDict[key]
            return classLabel

# 存储决策树
def storeTree(inputTree, fileName):
    file = open(fileName, "w")
    file.write(json.dumps(inputTree))
    file.close()

# 读取决策树
def readTree(fileName):
    file = open(fileName)
    fileStr = file.read()
    return json.loads(fileStr)

if __name__ == "__main__":
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    myTree = createTree(dataSet, labels.copy())
    classLabel = classify(myTree, labels, [1, 1])
    print(myTree)
    print(classLabel)
