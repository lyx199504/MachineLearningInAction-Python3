#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/17 21:33
# @Author : LYX-夜光

from ch12_fpGrowth import fpGrowth

if __name__ == "__main__":
    dataMat = [line.split() for line in open('kosarak.dat').readlines()]
    dataDict = fpGrowth.createInitSet(dataMat)
    myFPtree, myHeaderTab = fpGrowth.createTree(dataDict, 100000)
    myFreqList = []
    fpGrowth.mineTree(myHeaderTab, 100000, set([]), myFreqList)
    print(myFreqList)