#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/14 12:46
# @Author : LYX-夜光

from ch08_regression import regression


if __name__ == "__main__":
    abX, abY = regression.loadDataSet('abalone.txt')
    print("前100个数据加权回归：")
    yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    print("yHat0.1最小二乘：", regression.rssError(abY[0:99], yHat01.T))
    yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    print("yHat1最小二乘：", regression.rssError(abY[0:99], yHat1.T))
    yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print("yHat10最小二乘：", regression.rssError(abY[0:99], yHat10.T))

    print("前100个数据对后100个数据的加权回归预测：")
    yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("yHat0.1最小二乘：", regression.rssError(abY[100:199], yHat01.T))
    yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("yHat1最小二乘：", regression.rssError(abY[100:199], yHat1.T))
    yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("yHat10最小二乘：", regression.rssError(abY[100:199], yHat10.T))
