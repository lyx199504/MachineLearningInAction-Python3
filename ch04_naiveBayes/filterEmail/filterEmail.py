#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/6 19:03
# @Author : LYX-夜光

from ch04_naiveBayes import bayes
import re
import random

# 拆分字符串
def textParse(bigString):
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 测试邮件
def spamTest():
    docList, classList = [], []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)

    vocabList = bayes.createVocabList(docList)
    trainingSet, testSet = list(range(50)), []
    for i in range(10):  # 随机选择10个测试数据
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNaiveBayes(trainMat, trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNaiveBayes(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rete is: %f" % (float(errorCount)/len(testSet)))

if __name__ == "__main__":
    spamTest()

