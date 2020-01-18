#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/18 18:08
# @Author : LYX-夜光

import numpy as np

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print()
    print()

# 压缩图像
def imgCompress(numSV=3, thresh=0.8):
    # 参数numSV，代表压缩成多少维，numSV可根据Sigma来设定
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)  # 分解
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]  # 合并
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)

if __name__ == "__main__":
    imgCompress()