#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/16 21:27
# @Author : LYX-夜光

# 创建子集
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

# 计算子集出现的频率
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):  # can是tid的子集
                ssCnt[can] = ssCnt.get(can, 0) + 1  # 计数
    numItems = float(len(D))  # 数据组数
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems  # 每个元素出现的频率
        if support >= minSupport:
            retList.insert(0, key)  # 在第0个位置插入key
        supportData[key] = support
    return retList, supportData

# 合并新集合
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1, L2 = list(Lk[i])[:k-2], list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 初始子集（每个集合元素只有1个）
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)  # 根据支持度选取子集
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)  # 两两合并子集，合并后每个集合元素增加1个
        Lk, Sk = scanD(D, Ck, minSupport)  # 再根据支持度选取子集
        supportData.update(Sk)
        L.append(Lk)
        k += 1
    return L, supportData

# 计算关联
def calcConf(freqSet, H, supportData, bigRuleList, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]  # 集合支持率除以子集的支持率
        if conf >= minConf:
            bigRuleList.append((freqSet-conseq, conseq, conf))  # 关联集合表
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > m + 1:
        Hmp1 = aprioriGen(H, m+1)  # 两两组合元素多一个的新集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, bigRuleList, minConf)  # 筛选子集
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, bigRuleList, minConf)

# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

if __name__ == "__main__":
    dataSet = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    L, supportData = apriori(dataSet)
    bigRuleList = generateRules(L, supportData, minConf=0.5)
    for bigRule in bigRuleList:
        print(bigRule[0], "-->", bigRule[1], "conf:", bigRule[2])
