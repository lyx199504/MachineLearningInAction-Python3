#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/17 15:10
# @Author : LYX-夜光

from ch11_apriori import apriori

if __name__ == "__main__":
    # 数据集每一行的第一个数据为蘑菇的毒性：1无毒 2有毒，后面为蘑菇的特征
    mushDataMat = [line.split() for line in open('mushroom.dat').readlines()]
    L, supportData = apriori.apriori(mushDataMat, minSupport=0.3)
    for item in L[3]:  # 以3个特征为例，查看毒蘑菇的特征
        if '2' in item:
            print(item)