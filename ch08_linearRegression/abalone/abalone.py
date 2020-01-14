#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/14 12:46
# @Author : LYX-夜光

from ch08_linearRegression import regression

def abaloneByLwlr():
    abX, abY = regression.loadDataSet('abalone.txt')

    print("前100个数据加权线性回归：")
    yHat01 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    print("yHat0.1误差：", regression.rssError(abY[0:99], yHat01.T))
    yHat1 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    print("yHat1误差：", regression.rssError(abY[0:99], yHat1.T))
    yHat10 = regression.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print("yHat10误差：", regression.rssError(abY[0:99], yHat10.T))

    print("前100个数据对后100个数据的加权线性回归预测：")
    yHat01 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    print("yHat0.1误差：", regression.rssError(abY[100:199], yHat01.T))
    yHat1 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    print("yHat1误差：", regression.rssError(abY[100:199], yHat1.T))
    yHat10 = regression.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print("yHat10误差：", regression.rssError(abY[100:199], yHat10.T))

def abaloneByRidge():
    abX, abY = regression.loadDataSet('abalone.txt')
    ridgeWeights = regression.ridgeTest(abX, abY)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def abaloneByStage():
    abX, abY = regression.loadDataSet('abalone.txt')
    weights = regression.stageWise(abX, abY, 0.005, 1000)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(weights)
    plt.show()

if __name__ == "__main__":
     abaloneByStage()