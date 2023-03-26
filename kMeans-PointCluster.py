# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
import time


def loadDataSet(fileName):
    dataArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataArr.append(fltLine)
    return dataArr


#############################################################
# 输出向量A和向量B的欧式距离
# 根号(||A1-B1||^2+||A2-B2||^2+……+||An-Bn||^2)
#############################################################
def distEclud(vecA, vecB):
    euclidDistance = np.sqrt(np.sum(np.power(vecA - vecB, 2)))
    return euclidDistance


#############################################################
# 输入：dataSet, k
#      dataSet 样本数据集
#      k k个聚类
# 输出：centroids
#      centroids 每个簇的中心点
# 注释：随机初始化k个中心点
#############################################################
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  # 输入数据集dataSet有多少个特征变量
    centroids = np.mat(np.zeros((k, n)))  # 初始化k行n列的全0矩阵，用于存放簇心
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def nearest(point, cluster_centers, distMeas=distEclud, ):
    min_dist = np.inf
    m = np.shape(cluster_centers)[0]
    for i in range(m):
        d = distMeas(point, cluster_centers[i,])
        if min_dist > d:
            min_dist = d
    return min_dist


#############################################################
# def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent)
# 输入：dataSet, k, distMeas=distEclud, createCent=randCent
#      dataSet 样本数据集
#      k k个聚类
#      distMeas 距离计算方法，默认使用distEclud欧氏距离函数，属于函数的闭包用法
#      createCent 随机初始化k个簇的中心点，默认使用随机函数获取，属于函数的闭包用法
# 输出：centroids， clusterAssment
#      centroids 聚类结束后，每个簇的中心点坐标
#      clusterAssment
# 注释：kMeans聚类算法原理
#############################################################
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]  # 获取数据集样本数
    clusterAssment = np.mat(np.zeros((m, 2)))  # 初始化m行2列的全0矩阵
    centroids = createCent(dataSet, k)  # 随机创建初始的k个簇的中心点
    clusterChanged = True  # 聚类结果是否发生变化，二值变量（开关变量）
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历数据集的每一个样本
            minDist = float('inf')  # 初始化最小距离为正无穷大
            minIndex = -1  # 最小距离对应的索引为-1
            for j in range(k):  # 选定某个样本点，循环计算该样本分别到k个簇中心点的距离，存储最小距离及对应簇的中心点
                distJI = distMeas(centroids[j, :], np.mat(dataSet)[i, :])  # 计算数据点到簇中心点的欧氏距离
                if distJI < minDist:  # 如果距离小于当前最小距离
                    minDist = distJI  # 当前距离作为最小距离
                    minIndex = j  # 当前距离对应的第j个簇作为索引
            # 由于初始化clusterAssment为两列全0矩阵，因而第一次循环，二者肯定不会相等，之后循环，则会有簇的相同中心点不改变clusterChanged
            # 如果clusterAssment[i, 0] != minIndex，说明样本i绑定到其他簇的中心点，说明聚类没有完全收敛，继续执行聚类
            # 直到所有样本点均稳定绑定在其对应的簇的中心点上
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            # 更新clusterAssment中当前样本点i被绑定到簇的索引和样本点i到绑定簇中心点距离的平方
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):
            # 将数据集中所有属于当前簇的样本点通过条件过滤筛选
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 计算当前簇样本点的均值，作为新的簇中心点
            centroids[cent, :] = np.mean(ptsInClust, axis=0)

    return centroids, clusterAssment


#############################################################
# def plotDataSet(datMat, centList, clusterAssment, k)
# 输入：datMat, centList, clusterAssment, k
#      datMat 样本数据集
#      centList 聚类簇的中心点
#      clusterAssment 第一列：本点i被绑定到簇的索引，第二列：样本点i到绑定簇中心点距离的平方
#      k k个聚类
# 输出：
# 注释：kMeans聚类算法样本绘图
#############################################################
def plotDataSet(datMat, centList, clusterAssment, k):
    datMat = np.mat(datMat)
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']  # 样本聚类标记形状，共10种可选
    fig = plt.figure()
    ax = fig.add_axes(rect, label='ax', frameon=True)  # 添加绘图区域，图框开启
    # 绘制样本点
    for i in range(k):  # 循环遍历每个聚类
        ptsInCurrCluster = datMat[np.nonzero(clusterAssment[:, 0].A == i)[0], :]  # 筛选出属于i类的所有样本点
        markerStyle = scatterMarkers[i % len(scatterMarkers)]  # 选择聚类标记形状
        ax.scatter(ptsInCurrCluster[:, 0].flatten().A[0],
                   ptsInCurrCluster[:, 1].flatten().A[0],
                   marker=markerStyle,
                   s=90)  # 在坐标轴上绘制样本点，四个参数分别为样本点横坐标、纵坐标、样本点形状、样本点大小
    # 绘制质心
    for i in range(k):
        ax.scatter(centList[i].tolist()[0][0], centList[i].tolist()[0][1], s=100, c='k', marker='+', alpha=.5)
    plt.title('二分类k—Means聚类')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


###########################################################
#      实现k-Means聚类算法
############################################################
if __name__ == '__main__':
    startTime = time.time()
    dataset = loadDataSet('testSet.txt')
    dataset = np.mat(dataset)
    k = 4
    centList, clusterAssment = kMeans(dataset, k)  # 聚类算法
    plotDataSet(dataset, centList, clusterAssment, k)  # 绘图

    endTime = time.time()
    runTime = endTime - startTime
    print('程序运行时间为%d秒', runTime)
