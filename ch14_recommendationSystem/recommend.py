#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/18 14:31
# @Author : LYX-夜光

import numpy as np

# 欧几里得相似度
def ecludSim(inA, inB):
    return 1.0/(1.0 + np.linalg.norm(inA - inB))

# 皮尔森相关系数
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(inA, inB, rowvar=False)[0][1]

# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5*(num/denom)

# 基于物品相似度的推荐
def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal, ratSimTotal = 0.0, 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        # 被用户评分的两个物品
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])  # 两个物品相似度
        simTotal += similarity
        ratSimTotal += similarity * userRating
    return 0 if simTotal == 0 else ratSimTotal/simTotal  # 相似度加权计算item得分

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]  # 这个用户未评分的物品
    if len(unratedItems) == 0:
        return "所有商品已被该用户评分"
    itemsScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)  # 获取uesr对item的评分
        itemsScores.append((item, estimatedScore))
    return sorted(itemsScores, key=lambda x: x[1], reverse=True)[:N]

if __name__ == "__main__":
    dataMat = [[1, 1, 0, 2, 2],
               [0, 0, 0, 3, 3],
               [0, 0, 0, 1, 1],
               [1, 1, 1, 0, 0],
               [2, 2, 2, 0, 0],
               [1, 1, 1, 0, 0],
               [5, 5, 5, 0, 0]]
    myMat = np.mat(dataMat)
    # print(ecludSim(myMat[:, 0], myMat[:, 4]))
    # print(cosSim(myMat[:, 0], myMat[:, 4]))
    # print(pearsSim(myMat[:, 0], myMat[:, 4]))

    myMat[0, 0] = myMat[0, 1] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print(recommend(myMat, 2))