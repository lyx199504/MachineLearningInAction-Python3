#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/14 17:28
# @Author : LYX-夜光

from ch08_linearRegression import regression
import numpy as np
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib import rcParams

def getLegoDatas(retX, retY, setNum, year, numPce, origPrc):
    readFile = open("dataHtml/lego%s.txt" % setNum, 'rb')
    html = etree.HTML(readFile.read())
    tables = html.xpath('//div[@class="lview"]/table')
    for table in tables:
        tds = table.xpath('tr/td')
        span = tds[3].xpath('span')
        if not span or span[0].text != "Sold":  # 不出售
            continue
        newFlag = 0  # 是否新品
        nameStr = tds[1].xpath('div/a')[0].text
        if nameStr.lower().find("new") > -1:
            newFlag = 1
        priceStr = tds[4].text if tds[4].text else tds[4].xpath('div')[0].text
        sellingPrice = float(priceStr.replace('$', '').replace(',', ''))
        if sellingPrice > origPrc * 0.5:
            # print("%d\t%d\t%d\t%f\t%f" % (year, numPce, newFlag, origPrc, sellingPrice))
            retX.append([year, numPce, newFlag, origPrc])
            retY.append(sellingPrice)

def setDataCollect(retX, retY):
    getLegoDatas(retX, retY, 8288, 2006, 800, 49.99)
    getLegoDatas(retX, retY, 10030, 2002, 3096, 269.99)
    getLegoDatas(retX, retY, 10179, 2007, 5195, 499.99)
    getLegoDatas(retX, retY, 10181, 2007, 3428, 199.99)
    getLegoDatas(retX, retY, 10189, 2008, 5922, 299.99)
    getLegoDatas(retX, retY, 10196, 2009, 3263, 249.99)

if __name__ == "__main__":
    lgX, lgY = [], []
    setDataCollect(lgX, lgY)
    m, n = np.shape(lgX)
    lgX1 = np.mat(np.ones((m, n+1)))
    lgX1[:, 1:] = np.mat(lgX)

    rcParams['font.family'] = 'simhei'  # 显示中文
    plt.figure()
    # plt.subplot(111)
    line0, = plt.plot(lgY, label="原价格")
    # 最小二乘法预测
    weights = regression.standRegress(lgX1, lgY)
    line1, = plt.plot(lgX1 * weights, label="最小二乘预测")
    # 岭回归预测
    weights, b = regression.crossValidation(lgX, lgY)
    line2, = plt.plot(np.mat(lgX) * weights.T + b, label="岭回归预测")
    plt.legend(handles=[line0, line1, line2], loc='upper right')

    plt.show()
