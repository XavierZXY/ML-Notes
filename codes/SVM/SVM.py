#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : SVM.py
@Author      : Xavier ZXY
@Email       : zxy_xavier@163.com
@Date        : 2022/03/15 17:27
@Description : SVM支持向量机
"""


import time
import numpy as np
import math
import random


def loadData(fileName):
    """
    加载文件
    @Args:
        fileName: 文件路径

    @Returns:
        dataArr: 数据集
        labelArr: 数据集标签

    @Riase:

    """
    dataArr = []
    labelArr = []

    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        dataArr.append([int(num) / 255 for num in curLine[1:]])

        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)

    return dataArr, labelArr


class SVM:
    """
    SVM类
    """

    def __init__(self, trainDataList, trainLabelList, sigma=10, C=200, toler=0.001):
        """
        SVM相关参数初始化

        @Args:
            trainDataList: 训练数据集
            trainLabelList: 训练数据集标签
            sigma: 高斯核中分母的sigma, default=10
            C: 软间隔中的惩罚参数, default=200
            toler: 松弛变量, default=0.001

        @Returns:

        @Riase:

        """
        self.trainDataMat = np.mat(trainDataList)  # 训练数据集
        # 训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.trainLabelMat = np.mat(trainLabelList).T

        self.m, self.n = np.shape(self.trainDataMat)  # m：训练集数量    n：样本特征数目
        self.sigma = sigma  # 高斯核分母中的σ
        self.C = C  # 惩罚参数
        self.toler = toler  # 松弛变量

        self.k = self.calcKernel()  # 核函数（初始化时提前计算）
        self.b = 0  # SVM中的偏置b
        self.alpha = [0] * self.trainDataMat.shape[0]   # α 长度为训练集数目
        self.E = [0 * self.trainLabelMat[i, 0]
                  for i in range(self.trainLabelMat.shape[0])]  # SMO运算过程中的Ei
        self.supportVecIndex = []
