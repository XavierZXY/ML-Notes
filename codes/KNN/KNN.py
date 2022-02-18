#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : KNN.py
@Author      : Xavier ZXY
@Emial       : zxy_xavier@163.com
@Date        : 2022/02/18 15:57
@Description : K近邻算法
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

def calcDist(x1, x2):
    """
    计算两个样本点向量之间的距离
    使用的是欧氏距离
    @Args:
        x1: 向量1
        x2: 向量2
    
    @Returns:
        distance: 向量之间的欧氏距离
    
    @Riase:
    
    """

    return np.sqrt(np.sum(np.square(x1 - x2)))

def getClosest(trainDataMat, trainLabelMat, x, topK):
    """
    预测样本x的标记
    通过找到与x样本最近的topK个点，并查看它们的标签
    @Args:
        trainDataMat: 训练集数据
        trainLabelMat: 训练集标签
        x: 预测的样本x
        topK: 选择参考最邻近样本的数目
        
    @Returns:
        预测样本x的标记
    @Riase:
    
    """
    print("Start calculate distance...")
    # 建立一个列表存放向量x与训练集中每个样本的距离
    # 列表的长度为训练集的长度，distList[i]表示训练集中第i个样本与x的距离
    distList = [0] * len(trainLabelMat)
    # 遍历训练集中的所有样本点
    for i in range(len(trainDataMat)):
        # 获取训练集中当前样本的向量
        x1 = trainDataMat[i]
        # 计算向量x与训练集样本的距离
        curDist = calcDist(x1, x)
        distList[i] = curDist

    # 对距离列表进行升序排序
    topKList = np.argsort(np.array(distList))[:topK]
    # 投票表决，topK每人有一票
    labelList = [0] * 10
    # 对topK个索引进行遍历
    for index in topKList:
        # trainLabelMat[index]: 寻找topK元素索引对应的标记
        # int(trainLabelMat[index]): 转换标记为int
        # labelList(int(trainLabelMat[index])): 标记labelList对应的位置
        labelList[int(trainLabelMat[index])] += 1

    # 返回选票最多的票数值
    return labelList.index(max(labelList))

def model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):
    """
    测试模型
    @Args:
        trainDataArr: 训练集数据
        trainLabelArr: 训练集标记
        testDataArr: 测试集数据
        testLabelArr: 测试集标记
        topK: 选择多少个邻近点

    @Returns:
        正确率

    @Riase:
    
    """
    print("Start test...")
    # 将所有列表转换为矩阵形式
    trainDataMat = np.mat(trainDataArr)
    trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr)
    testLabelMat = np.mat(testLabelArr).T

    # 错误值技术
    errorCnt = 0
    # 遍历测试集，对每个测试集样本进行测试
    for i in range(len(testDataMat)):
        print("test {}:{}".format(i, len(testDataMat)))
        # 读取测试集当前测试样本
        x = testDataMat[i]
        # 获取预测的标记
        y = getClosest(trainDataMat, trainLabelMat, x, topK)
        # 预测不符
        if y != testLabelMat[i]:
            errorCnt += 1

    # 返回正确率
    return 1 - (errorCnt / len(testDataMat))

if __name__ == "__main__":
    start = time.time()

    #获取训练集
    trainDataArr, trainLabelArr = loadData('../data/mnist_train.csv')
    #获取测试集
    testDataArr, testLabelArr = loadData('../data/mnist_test.csv')
    #计算测试集正确率
    accur = model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    #打印正确率
    print("accur is:{}".format(accur * 100), "%")

    end = time.time()
    #显示花费时间
    print('time span:', end - start)