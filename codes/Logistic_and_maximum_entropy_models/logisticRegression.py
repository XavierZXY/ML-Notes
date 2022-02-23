#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : logisticRegression.py
@Author      : Xavier ZXY
@Emial       : zxy_xavier@163.com
@Date        : 2022/02/23 16:35
@Description : 逻辑斯蒂回归
"""

import time
import numpy as np


def loadData(fileName):
    """
    加载数据

    @Args:
        fileName: 需要加载的文件
    
    @Returns:
        dataList: 数据集
        labelLise: 标签集
    
    @Riase:
    
    """

    dataList = []
    labelList = []
    # 打开文件
    fr = open(fileName, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        
        if int(curLine[0]) == 0:
            labelList.append(1)
        else:
            labelList.append(0)

        dataList.append([int(num) / 255 for num in curLine[1:]])

    return dataList, labelList

def predict(w, x):
    """
    预测标签

    @Args:
        w: 权重
        x: 样本
    
    @Returns:
        预测结果
    
    @Riase:
    
    """

    wx = np.dot(w, x)
    P1 = np.exp(wx) / (1 + np.exp(wx))
    if P1 >= 0.5:
        return 1
    
    return 0

def logisticRegression(trainDataList, trainLabelList, iter=20):
    """
    逻辑斯蒂回归训练

    @Args:
        trainDataList: 训练集
        trainLabelList: 训练集标签
        iter: 迭代次数, default=20
    @Returns:
        w: 训练得到的权重
    
    @Riase:
    
    """

    # 将w与b合在一起，x需要增加一维
    for i in range(len(trainDataList)):
        trainDataList[i].append(1)

    trainDataList = np.array(trainDataList)
    w = np.zeros(trainDataList.shape[1])

    # 设置步长
    h = 0.001

    # 迭代iter次进行随机梯度下降
    for i in range(iter):
        # 每次迭代遍历所有样本，进行随机梯度下降
        print(f"Epoch: {i}, remaining {iter - i} ......")
        for j in range(trainDataList.shape[0]):
            # 随机梯度上升部分
            # 我们需要极大化似然函数但是似然函数由于有求和项，
            # 并不能直接对w求导得出最优w，所以针对似然函数求和部分中每一项进行单独地求导w，
            # 得到针对该样本的梯度，并进行梯度上升（因为是要求似然函数的极大值，
            # 所以是梯度上升，如果是极小值就梯度下降。梯度上升是加号，下降是减号）
            # 求和式中每一项单独对w求导结果为：xi * yi - (exp(w * xi) * xi) / (1 + exp(w * xi))
            wx = np.dot(w, trainDataList[j])
            xi = trainDataList[j]
            yi = trainLabelList[j]
            # 梯度上升
            w +=  h * (xi * yi - (np.exp(wx) * xi) / ( 1 + np.exp(wx)))

    return w

def model_test(testDataList, testLabelList, w):
    """
    模型测试

    @Args:
        testDataList: 测试数据集
        testLabelList: 测试数据集标签
        w: 学习到的权重
    
    @Returns:
        准确率
    
    @Riase:
    
    """

    for i in range(len(testDataList)):
        testDataList[i].append(1)

    # 错误值计数
    errorCnt = 0
    for i in range(len(testDataList)):
        if testLabelList[i] != predict(w, testDataList[i]):
            errorCnt += 1
    
    # 返回正确率
    return 1 - errorCnt / len(testDataList)

if __name__ == '__main__':
    
    start = time.time()

    # 获取训练集
    print("Start read transSet......")
    trainData, trainLabel = loadData("../data/mnist_train.csv")

    # 获取测试集
    print("Start read testSet......")
    testData, testLabel = loadData("../data/mnist_test.csv")

    # 开始训练
    print("Start to train......")
    w = logisticRegression(trainData, trainLabel)

    # 验证正确率
    print("Start to test......")
    accuracy = model_test(testData, testLabel, w)
    print(f"The accuracy is: {accuracy}")

    end = time.time()
    print(f"Time span: {end - start}")