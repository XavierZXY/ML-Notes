from base64 import b16decode
from cProfile import label
from email import header
from re import I
from tkinter.tix import MAIN
from pip import main

from sklearn.metrics import label_ranking_average_precision_score

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : perceptron_dichotomy.py
@Author      : Xavier ZXY
@Emial       : zxy_xavier@163.com
@Date        : 2022/02/16 16:17
@Description : 感知机算法实现
"""

import numpy as np
import time

def loadData(fileName):
    """
    
    @Args:
        fileName: 需要加载的数据集路径
    
    @Returns:
        dataArr: list形式的数据集
        labelArr: list形式的标签
    
    @Riase:
    
    """

    print("Start to read data...")
    # 定义dataArr和lableArr
    dataArr = []
    lableArr = []
    # 打开文件
    fr = open(fileName, 'r')
    # 文件按行进行读取
    for line in fr.readlines():
        # 对每一行数据按切割符','进行切割，返回字段列表
        curLine = line.strip().split(',')
        # Mnist有0-9个标记，由于是二分类任务，所以将>=5的作为1，<5的作为-1
        if int(curLine[0])>= 5:
            lableArr.append(1)
        else:
            lableArr.append(-1)
        # 将除第一行标记数据外的所有数据进行归一化
        dataArr.append([int(num)/255 for num in curLine[1:]])

    # 返回data和label
    return dataArr, lableArr

def perceptron(dataArr, labelArr, iter=50):
    """
    感知机训练
    @Args:
        dataArr:训练集数据
        labelArr: 训练集标签
        iter: 迭代次数，默认50
    
    @Returns:
        w, b: 训练后的权重
    
    @Riase:
    
    """

    print("Start to trains...")
    # 将数据转换为矩阵形式
    # 转换后的数据中每一个样本的向量都是横向的
    dataMat = np.mat(dataArr)
    # 将标签转换为矩阵，之后转置
    labelMat = np.mat(labelArr).T
    # 获取数据矩阵的大小，为m*n
    m, n = np.shape(dataMat)
    # 初始化权重w，初始值全为0
    w = np.zeros((1, np.shape(dataMat)[1]))
    # 初始化偏置b为0
    b = 0
    # 初始化补偿，即梯度下降过程中的n，控制梯度下降速率
    h = 0.0001

    # 进行iter次迭代计算
    for k in range(iter):
        for i in range(m):
            # 随机梯度下降
            xi = dataMat[i]
            # 获取当前样本所对应的标签
            yi = labelMat[i]
            # 判断是否为误分类样本
            # 误分类样本特征：-yi(w*xi.T + b) >= 0
            if -1 * yi * (w * xi.T + b) >= 0:
                # 对于误分类样本，进行梯度下降，更新w和b
                w = w + h * yi * xi
                b = b + h * yi
        # 打印训练进度
        print("Round %d:%d training" % (k, iter))

    # 返回训练完的参数
    return w, b

def model_test(dataArr, labelArr, w, b):
    """
    测试准确率
    @Args:
        dataArr:训练集数据
        labelArr: 训练集标签
        iter: 迭代次数，默认50
        w: 训练得到的权重
        b: 训练得到的偏置
    
    @Returns:
        accruRate: 准确率
    
    @Riase:
    
    """

    print("Start to test...")
    # 将数据集转换为矩阵形式
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    # 获取测试数据集矩阵的大小
    m, n = np.shape(dataMat)
    # 错误样本数基数
    errorCnt = 0
    # 遍历所有测试样本
    for i in range(m):
        # 获得单个样本向量
        xi = dataMat[i]
        # 获得该样本标记
        yi = labelMat[i]
        # 获得运算结果
        result = -1 * yi * (w * xi.T + b)
        if result >= 0: errorCnt += 1
    # 正确率
    accruRate = 1 - (errorCnt / m)

    return accruRate

if __name__ == '__main__':
    # 获取当前时间
    start = time.time()

    # 获取训练集及标签
    trainData, trainLabel = loadData('../data/mnist_train.csv')
    # 获取测试及标签
    testData, testLabel = loadData('../data/mnist_train.csv')
    
    #训练获得权重
    w, b = perceptron(trainData, trainLabel, iter = 30)
    #进行测试，获得正确率
    accruRate = model_test(testData, testLabel, w, b)

    #获取当前时间，作为结束时间
    end = time.time()
    #显示正确率
    print('accuracy rate is:', accruRate)
    #显示用时时长
    print('time span:', end - start)