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

if __name__ == "__main__":
    dataSet = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    L, supportData = apriori(dataSet)
    print(L)

