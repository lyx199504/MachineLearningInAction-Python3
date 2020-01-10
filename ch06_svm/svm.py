#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/11 0:33
# @Author : LYX-夜光

import numpy as np

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn  # 数据矩阵
        self.y = classLabels  # 类别：1或-1
        self.C = C  # 最大间隔：0 < C < 1
        self.tol = toler  # 容忍度：0附近的正数
        self.m = np.shape(dataMatIn)[0]  # m组数据
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 拉格朗日乘数法参数
        self.b = 0  # 拟合函数y = (W^T)*X + b的截距
        self.eCache = np.mat(np.zeros((self.m, 2)))

    def calcEk(self, k):
        fXk = float((np.multiply(self.alphas, self.y)).T * (self.X*self.X[k, :].T)) + self.b
        