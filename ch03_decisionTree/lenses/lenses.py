#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/4 22:57
# @Author : LYX-夜光

from ch03_decisionTree import decisionTree

# 初始化数据
def file2dataSet(fileName):
    readFile = open(fileName)
    arrayOfLines = readFile.readlines()
    dataSet = []
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        dataSet.append(listFromLine)
    return dataSet

# 创建隐形眼镜决策树
def createLensesTree():
    dataSet = file2dataSet("lenses.txt")
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 标签是：年龄，处方，散光，流泪程度?
    myTree = decisionTree.createTree(dataSet, lensesLabels)
    return myTree

if __name__ == "__main__":
    myTree = createLensesTree()
    print(myTree)

    # # 测试样例
    # lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # testVec = ['presbyopic', 'myope', 'no', 'normal']
    # print(decisionTree.classify(myTree, lensesLabels, testVec))