#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : DecisionTree.py
@Author      : Xavier ZXY
@Emial       : zxy_xavier@163.com
@Date        : 2022/02/21 16:59
@Description : 决策树
"""

import time
import numpy as np

def loadData(fileName):
    """
    加载文件
    @Args:
        fileName: 加载的文件路径
    
    @Returns:
        dataArr: 数据集
        labelArr: 标签集
    
    @Riase:
    
    """
    
    # 存放数据和标记
    dataArr = []
    labelArr = []
    # 读取文件
    fr = open(fileName)
    # 遍历文件中的每一行
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # 将数据进行了二值化处理
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        # 标签
        labelArr.append(int(curLine[0]))

    return dataArr, labelArr

def majorClass(labelArr):
    """
    找到当前标签集中数目最多的标签

    @Args:
        labelArr: 标签集

    @Returns:
        出现次数最多的标签

    @Riase:
    
    """

    # 存储不同类别标签数目
    classDict = {}
    # 遍历所有标签
    maxtimes = -1
    maxClass = 0
    for i in range(len(labelArr)):
        if labelArr[i] in classDict.keys():
            classDict[labelArr[i]] += 1
        else:
            classDict[labelArr[i]] = 1
        # 只找出现次数最多的标签
        if classDict[labelArr[i]] > maxtimes:
            maxtimes = classDict[labelArr[i]]
            maxClass = labelArr[i]

    # 降序排序
    # classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)

    # 返回出现次数最多的标签
    return maxClass
    # return classSort[0][0]

def calc_H_D(trainLabelArr):
    """
    计算数据集D的经验熵
    @Args:
        trainLabelArr: 数据集的标签集
    
    @Returns:
        H_D: 经验熵
    @Riase:
    
    """

    # 初始化H_D
    H_D = 0
    # 使用集合处理trainLabelSet
    trainLabelSet = set([label for label in trainLabelArr])
    # 遍历
    for i in trainLabelSet:
        # 计算 |Ck|/|D|
        p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size
        # 经验熵累加求和
        H_D += -1 * p * np.log2(p)

    return H_D

def calcH_D_A(trainDataArr_DevFeature, trianLabelArr):
    """
    计算经验条件熵

    @Args:
        trainDataArr_DevFeature: 切割后只有feature那列数据的数组
        trianLabelArr: 标签集

    @Returns:
        H_D_A: 经验条件熵

    @Riase:
    
    """

    # 初始化H_D_A
    H_D_A = 0
    trainDataSet = set([label for label in trainDataArr_DevFeature])
    # 遍历特征
    for i in trainDataSet:
        # 计算H(D|A)
        H_D_A += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size \
                    * calc_H_D(trianLabelArr[trainDataArr_DevFeature == i]) 

    return H_D_A

def calcBestFeature(trainDataList, trainLabelList):
    """
    计算信息增益最大的特征

    @Args:
        trainDataList: 数据集
        trainLabelList: 数据集标签
    
    @Returns:
        maxFeature: 信息增益最大的特征
        maxG_D_A: 最大信息增益值
    @Riase:
    
    """

    trainDataArr = np.array(trainDataList)
    trainLabelArr = np.array(trainLabelList)

    # 获取特征数目
    featureNum = trainDataArr.shape[1]

    # 初始化最大信息增益
    maxG_D_A = -1
    # 初始化最大信息增益的特征
    maxFeature = -1

    # 计算经验熵
    H_D = calc_H_D(trainLabelArr)

    # 遍历每一个特征
    for feature in range(featureNum):
        # 计算条件熵H(D|A)
        # 选取一列特征
        trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)
        # 计算信息增益G(D|A)
        G_D_A = H_D - calcH_D_A(trainDataArr_DevideByFeature, trainLabelArr)
        # 更新最大的信息增益以及对应的feature
        if G_D_A > maxG_D_A:
            maxG_D_A = G_D_A
            maxFeature = feature

    return maxFeature, maxG_D_A

def getSubDataArr(trainDataArr, trainLabelArr, A, a):
    """
    更新数据集和标签集
    @Args:
        trainDataArr: 要更新的数据集
        trainLabelArr: 要更新的标签集
        A: 要去除的特征索引
        a: 当data[A]== a时，说明该行样本时要保留的
    
    @Returns:
        retDataArr: 新的数据集
        retLabelArr: 新的标签集
    @Riase:
    
    """

    retDataArr = []
    retLabelArr = []

    # 对当前数据的每一个样本进行遍历
    for i in range(len(trainDataArr)):
        # 如果当前样本的特征为指定特征值a
        if trainDataArr[i][A] == a:
            # 那么将该样本的第A个特征切割掉，放入返回的数据集中
            retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A+1:])
            #将该样本的标签放入返回标签集中
            retLabelArr.append(trainLabelArr[i])
    
    # 返回新的数据集和标签集
    return retDataArr, retLabelArr

def createTree(*dataSet):
    """
    递归创建决策树

    @Args:
        dataSet: (trainDataList, trainLabelList)

    @Returns:
        treeDict: 树节点

    @Riase:
    
    """

    # 设置阈值Epsilon
    Epsilon = 0.1
    trainDataList = dataSet[0][0]
    trainLabelList = dataSet[0][1]

    print("Start a node...", len(trainDataList[0]), len(trainLabelList))

    classDict = {i for i in trainLabelList}

    # 数据集D中所有实例属于同一类
    if len(classDict) == 1:
        return trainLabelList[0]

    # 如果A为空集，则置T为单节点数，并将D中实例数最大的类Ck作为该节点的类，返回T
    # 即如果已经没有特征可以用来再分化了，就返回占大多数的类别
    if len(trainDataList[0]) == 0:
        return majorClass(trainLabelList)

    # 否则，选择信息增益最大的特征
    Ag, EpsilonGet = calcBestFeature(trainDataList, trainLabelList)

    # 信息增益小于阈值Epsilon，置为单节点树
    if EpsilonGet < Epsilon:
        return majorClass(trainLabelList)
    
    # 否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，将Di中实例数最大的
    # 类作为标记，构建子节点，由节点及其子节点构成树T，返回T
    treeDict = {Ag:{}}
    # 特征值为0时，进入0分支
    # getSubDataArr(trainDataList, trainLabelList, Ag, 0)：在当前数据集中切割当前feature，返回新的数据集和标签集
    treeDict[Ag][0] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 0))
    treeDict[Ag][1] = createTree(getSubDataArr(trainDataList, trainLabelList, Ag, 1))

    return treeDict

def predict(testDataList, tree):
    """
    预测标签

    @Args:
        testDataList: 测试集
        tree: 决策树

    @Returns:
        预测结果

    @Riase:
    
    """

    # 死循环，直到找到一个有效地分类
    while True:
        # 因为有时候当前字典只有一个节点
        # 例如{73: {0: {74:6}}}看起来节点很多，但是对于字典的最顶层来说，只有73一个key，其余都是value
        # 若还是采用for来读取的话不太合适，所以使用下行这种方式读取key和value
        (key, value), = tree.items()
        
        if type(tree[key]).__name__ == "dict":
            dataVal = testDataList[key]
            del testDataList[key]
            # 将tree更新为其子节点的字典
            tree = value[dataVal]

            if type(tree).__name__ == "int":
                return tree
        else:
            return value

def model_test(testDataList, testLabelList, tree):
    """
    测试准确率

    @Args:
        testDataList: 测试数据集
        testLabelList: 测试数据集标签
        tree: 决策树

    @Returns:
        准确率
    @Riase:
    
    """

    # 错误判断次数
    errorCnt = 0
    #遍历测试集中每一个测试样本
    for i in range(len(testDataList)):
        #判断预测与标签中结果是否一致
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    #返回准确率
    return 1 - errorCnt / len(testDataList)

if __name__ == '__main__':

    # 开始时间
    start = time.time()

    # 获取训练集
    print("Start read transSet......")
    trainDataList, trainLabelList = loadData('../data/mnist_train.csv')
    # 获取测试集
    print("Start read testSet......")
    testDataList, testLabelList = loadData('../data/mnist_test.csv')

    # 创建决策树
    print("Start create tree......")
    tree = createTree((trainDataList, trainLabelList))
    print("tree is:", tree)

    # 测试准确率
    print("Start test......")
    accur = model_test(testDataList, testLabelList, tree)
    print("The accur is:{}".format(accur))

    # 结束时间
    end = time.time()
    print("Time span: {}".format(end - start))