#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/5 10:24
# @Author : LYX-夜光

import numpy as np

# 创建不重复的单词表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet |= set(document)
    return list(vocabSet)

# 记录单词表中出现过的单词的次数
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# 朴素贝叶斯分类器训练
def trainNaiveBayes(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 文档数
    numWords = len(trainMatrix[0])  # 特征单词的个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 侮辱性文档在总文档中占比
    p1Num, p0Num = np.ones(numWords), np.ones(numWords)  # 结果需要取对数，防止分量为0，因此初始化为1
    p1Denom, p0Denom = 2.0, 2.0  # 防止取对数后值为0，因此需要pxDenom > pxNum，故初始化为2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 侮辱性文档
            p1Num += trainMatrix[i]  # 侮辱性文档中存在某单词的文档数
            p1Denom += sum(trainMatrix[i])  # 侮辱性文档中单词总和
        else:  # 正常文档（与侮辱性文档类似）
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # 侮辱性文档中存在某单词的概率，并转化为对数
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类
def classifyNaiveBayes(vec2classify, p0Vec, p1Vec, pClass1):
    # 贝叶斯公式：P(c|v) = P(c)*P(v|c)/P(v)
    # 其中v为单词向量，c为分类：0正常，1侮辱性
    # 由于P(v)为常数，因此计算P(c)*P(v|c)即可，而后比较P(c|v)大小
    p1 = sum(vec2classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2classify * p0Vec) + np.log(1.0 - pClass1)
    return 1 if p1 > p0 else 0

if __name__ == "__main__":
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 每个子list为一个文档
    classVec = [0, 1, 0, 1, 0, 1]  # 0正常言论 1侮辱性言论
    vocabList = createVocabList(postingList)
    trainMat = []
    for post in postingList:
        trainMat.append(bagOfWords2Vec(vocabList, post))
    p0V, p1V, pAb = trainNaiveBayes(trainMat, classVec)

    testEntry = ['love', 'my', 'delmation']
    thisDoc = np.array(bagOfWords2Vec(vocabList, testEntry))
    print(testEntry, "classified as: ", classifyNaiveBayes(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(bagOfWords2Vec(vocabList, testEntry))
    print(testEntry, "classified as: ", classifyNaiveBayes(thisDoc, p0V, p1V, pAb))