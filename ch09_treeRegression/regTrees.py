#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/15 14:55
# @Author : LYX-夜光

import numpy as np

# 初始化数据
def loadDataSet(fileName):
    dataMat = []
    readFile = open(fileName)
    for line in readFile.readlines():
        fltLine = []
        curLine = line.strip().split('\t')
        for data in curLine:
            fltLine.append(float(data))   # 将每行映射成浮点数
        dataMat.append(fltLine)
    return dataMat

# 以某特征与某数值的大小关系来区分数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

# 数据最后一列的均值EX
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

# 数据最后一列的数据量×方差nDX
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

# 选择最好的划分
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS, tolN = ops
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # dataSet最后一列的不重复数据量
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS, bestIndex, bestValue = np.inf, 0, 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:  # mat0或mat1的数据量小于tolN
                continue
            newS = errType(mat0) + errType(mat1)
            if bestS > newS:  # 取最小的方差S
                bestS, bestIndex, bestValue = newS, featIndex, splitVal
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue

# 创建回归树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)  # 右子树
    return retTree

# 判断是否为树
def isTree(obj):
    return type(obj).__name__ == 'dict'

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0:  # 空数据
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        return treeMean if errorMerge < errorNoMerge else tree
    return tree

if __name__ == "__main__":
    # dataSet = loadDataSet('ex00.txt')
    # dataSet = np.mat(dataSet)
    # print(createTree(dataSet))

    # dataSet = loadDataSet('ex0.txt')
    # dataSet = np.mat(dataSet)
    # print(createTree(dataSet))

    myDat2 = loadDataSet('ex2.txt')
    myMat2 = np.mat(myDat2)
    myTree = createTree(myMat2, ops=(0, 1))
    print("剪枝前：", myTree)
    myDat2Test = loadDataSet('ex2test.txt')
    myMat2Test = np.mat(myDat2Test)
    myTree = prune(myTree, myMat2Test)
    print("剪枝后：", myTree)