#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/11 0:33
# @Author : LYX-夜光

from ch06_svm import svmSimplify
import numpy as np
import random

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = np.mat(dataMatIn)  # 数据矩阵
        self.y = np.mat(classLabels).T  # 类别：1或-1
        self.C = C  # 最大间隔：0 < C < 1
        self.tol = toler  # 容忍度：0附近的正数
        self.m = np.shape(dataMatIn)[0]  # m组数据
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 拉格朗日乘数法参数
        self.b = 0  # 拟合函数y = (W^T)*X + b的截距
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = self.kernelTrans(self.X, self.X[i, :], kTup)

    def kernelTrans(self, X, A, kTup):
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        if kTup[0] == 'lin':
            K = X * A.T
        elif kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow*deltaRow.T
            K = np.exp(K/(-1*kTup[1]**2))
        else:
            raise NameError('Houston We Have a Problem That Kernel is not recognized')
        return K

    # 计算误差
    def calcEk(self, k):
        # fXk = float(np.multiply(self.alphas, self.y).T * (self.X*self.X[k, :].T) + self.b)
        fXk = float(np.multiply(self.alphas, self.y).T * self.K[:, k] + self.b)
        Ek = fXk - float(self.y[k])
        return Ek

    # 调整大于H或小于L的alpha值
    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj

    # 从0-m内随机选择一个不同于i的下标j
    def selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    # 选择具有最大步长的j
    def selectJ(self, i, Ei):
        maxK, maxDeltaE, Ej = -1, 0, 0
        self.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(self.eCache[:, 0].A)[0]  # 非0值索引中的行索引
        if len(validEcacheList) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if maxDeltaE < deltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = self.selectJrand(i, self.m)
            Ej = self.calcEk(j)
            return j, Ej

    def updataEk(self, k):
        Ek = self.calcEk(k)
        self.eCache[k] = [1, Ek]

    def innerL(self, i):
        Ei = self.calcEk(i)
        if self.y[i]*Ei < -self.tol and self.alphas[i] < self.C or self.y[i]*Ei > self.tol and self.alphas[i] > 0:
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if self.y[i] != self.y[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                return 0
            # eta = 2.0 * self.X[i, :]*self.X[j, :].T - self.X[i, :]*self.X[i, :].T - self.X[j, :]*self.X[j, :].T
            eta = 2.0 * self.K[i, j] - self.K[i, j] - self.K[j, j]
            if eta >= 0:
                return 0
            self.alphas[j] -= self.y[j]*(Ei - Ej)/eta
            self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)
            self.updataEk(j)
            if abs(self.alphas[j] - alphaJold) < 0.00001:
                return 0
            self.alphas[i] += self.y[j]*self.y[i]*(alphaJold - self.alphas[j])
            self.updataEk(i)
            # b1 = self.b - Ei - self.y[i]*(self.alphas[i]-alphaIold)*self.X[i, :]*self.X[i, :].T -\
            #      self.y[j]*(self.alphas[j]-alphaJold)*self.X[i, :]*self.X[j, :].T
            # b2 = self.b - Ej - self.y[i]*(self.alphas[i]-alphaIold)*self.X[i, :]*self.X[j, :].T -\
            #      self.y[j]*(self.alphas[j]-alphaJold)*self.X[j, :]*self.X[j, :].T
            b1 = self.b - Ei - self.y[i] * (self.alphas[i] - alphaIold) * self.K[i, i] - \
                 self.y[j] * (self.alphas[j] - alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.y[i] * (self.alphas[i] - alphaIold) * self.K[i, j] - \
                 self.y[j] * (self.alphas[j] - alphaJold) * self.K[j, j]
            if 0 < self.alphas[i] < self.C:
                self.b = b1
            elif 0 < self.alphas[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2)/2.0
            return 1
        return 0

    def smoP(self, maxIter):
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(self.m):
                    alphaPairsChanged += self.innerL(i)
                iter += 1
            else:
                nonBoundIs = np.nonzero((self.alphas.A > 0)*(self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
            print("iteration number: %d" % iter)


if __name__ == "__main__":
    # 线性分隔平面
    dataMat, labelMat = svmSimplify.loadDataSet("testSet.txt")
    opt = optStruct(dataMat, labelMat, 0.6, 0.001, ('lin', 0))
    opt.smoP(100)
    svmSimplify.plotFit(dataMat, labelMat, opt.alphas, opt.b)

    # # 非线性分类
    # dataMat, labelMat = svmSimplify.loadDataSet("testSetRBF.txt")
    # opt = optStruct(dataMat, labelMat, 200, 0.0001, ('rbf', 1.3))
    # opt.smoP(10000)
    # index = np.nonzero(opt.alphas.A > 0)[0]
    # X, y = opt.X[index], opt.y[index]
    # errorCount = 0
    # for i in range(opt.m):
    #     kernelEval = opt.kernelTrans(X, opt.X[i, :], ('rbf', 1.3))
    #     predict = kernelEval.T * np.multiply(y, opt.alphas[index]) + opt.b
    #     if np.sign(predict) != np.sign(labelMat[i]):
    #         errorCount += 1
    # print("the training error rate is: %f" % (float(errorCount)/opt.m))
    #
    # dataMat, labelMat = svmSimplify.loadDataSet("testSetRBF2.txt")
    # errorCount = 0
    # m = np.shape(dataMat)[0]
    # for i in range(m):
    #     kernelEval = opt.kernelTrans(X, np.array(dataMat)[i, :], ('rbf', 1.3))
    #     predict = kernelEval.T * np.multiply(y, opt.alphas[index]) + opt.b
    #     if np.sign(predict) != np.sign(labelMat[i]):
    #         errorCount += 1
    # print("the test error rate is: %f" % (float(errorCount) / m))