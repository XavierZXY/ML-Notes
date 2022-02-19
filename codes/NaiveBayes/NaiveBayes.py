#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : NaiveBayes.py
@Author      : Xavier ZXY
@Emial       : zxy_xavier@163.com
@Date        : 2022/02/19 16:52
@Description : 朴素贝叶斯
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
    
    # 存放数据及标记
    dataArr = [] 
    labelArr = []
    # 读取文件
    fr = open(fileName)
    # 遍历文件中的每一行
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # 此外将数据进行了二值化处理，大于128的转换成1，小于的转换成0，方便后续计算
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        # 将标记信息放入标记集中
        # 放入的同时将标记转换为整型
        labelArr.append(int(curLine[0]))
    # 返回数据集和标记
    return dataArr, labelArr

def NaiveBayes(Py, Px_y, x):
    """
    通过朴素贝叶斯进行概率估计
    @Args:
        Py: 先验概率分布
        Px_y: 条件概率分布
        x: 要估计的样本x
    
    @Returns:
        所有label的估计概率
    
    @Riase:
    
    """

    # 设置特征数目
    featureNum = 784
    # 设置类别数目
    classNum = 10
    # 建立存放所有标记的估计概率数组
    P = [0] * classNum
    # 对于每个类别进行估计
    for i in range(classNum):
        # 初始化sum
        # 使用log对概率进行处理，概率连乘变连加
        sum = 0
        # 获取每一个条件概率值
        for j in range(featureNum):
            sum += Px_y[i][j][int(x[j])]
        
        # 和先验概率相加
        P[i] = sum + Py[i]
    
    return P.index(max(P))

def getAllProbability(trainDataArr, trainLabelArr):
    """
    通过训练集计算先验概率分布和条件概率分布
    @Args:
        trainDataArr: 训练集数据
        trainLabelArr: 训练集标签

    @Returns:
        Py: 先验概率分布
        Px_y: 条件概率分布

    @Riase:
    
    """

    # 手写图片28*28，转换为一维向量为784维
    featureNum = 784
    # 类别数目
    classNum = 10
    # 初始化先验概率分布数组
    Py = np.zeros((classNum, 1))

    # 对每个类别计算它的先验概率分布
    for i in range(classNum):
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)

    Py = np.log(Py)

    # 计算条件概率分布 Px_y = P(X=x|Y=y)
    Px_y = np.zeros((classNum, featureNum, 2))
    # 对标记遍历
    for i in range(len(trainLabelArr)):
        label = trainLabelArr[i]
        x = trainDataArr[i]
        # 对样本的每一维特征遍历
        for j in range(featureNum):
            Px_y[label][j][int(x[j])] += 1

    for label in range(classNum):
        for j in range(featureNum):
            # 获取y=label，第j个特征为0的个数
            Px_y0 = Px_y[label][j][0]
            # 获取y=label，第j个特征为1的个数
            Px_y1 = Px_y[label][j][1]

            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    return Py, Px_y

def model_test(Py, Px_y, testDataArr, testLabelArr):
    """
    模型测试
    @Args:
        Py: 先验概率分布
        Px_y: 条件概率分布
        testDataArr: 测试集数据
        testLabelArr: 测试集标签
    
    @Returns:
        准确率
    
    @Riase:
    
    """

    # 错误值
    errorCnt = 0
    # 循环遍历测试集中的每一个样本
    for i in range(len(testDataArr)):
        # 获取预测值
        presict = NaiveBayes(Py, Px_y, testDataArr[i])
        if presict != testLabelArr[i]:
            errorCnt += 1
    # 返回准确率
    return 1 - (errorCnt / len(testDataArr))

if __name__ == "__main__":

    start = time.time()
    # 获取训练集
    print("Start read transSet......")
    trainDataArr, trainLabelArr = loadData('../data/mnist_train.csv')

    # 获取测试集
    print("Start read testSet......")
    testDataArr, testLabelArr = loadData('../data/mnist_test.csv')

    #开始训练，学习先验概率分布和条件概率分布
    print("Start to train......")
    Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)

    #使用习得的先验概率分布和条件概率分布对测试集进行测试
    print("Start to test......")
    accuracy = model_test(Py, Px_y, testDataArr, testLabelArr)

    #打印准确率
    print("the accuracy is: {}".format(accuracy))
    #打印时间
    print("time span: {}".format(time.time() -start))