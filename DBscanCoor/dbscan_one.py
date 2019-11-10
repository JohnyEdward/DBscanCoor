
# -*- coding: utf-8 -*-
import numpy as np
import math


def distance(data):
    '''计算样本点之间的距离
    :param data(mat):样本
    :return:dis(mat):样本点之间的距离
    '''
    m, n = np.shape(data)
    dis = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(i, m):
            # 计算i和j之间的欧式距离
            tmp = 0
            for k in range(n):
                tmp += (data[i, k] - data[j, k]) * (data[i, k] - data[j, k])
            dis[i, j] = np.sqrt(tmp)
            dis[j, i] = dis[i, j]
    return dis


def find_eps(distance_D, eps):
    '''找到距离≤eps的样本的索引
    :param distance_D(mat):样本i与其他样本之间的距离
    :param eps(float):半径的大小
    :return: ind(list):与样本i之间的距离≤eps的样本的索引
    '''
    ind = []
    n = np.shape(distance_D)[1]
    for j in range(n):
        if distance_D[0, j] <= eps:
            ind.append(j)
    return ind


def dbscan(data, eps, MinPts):
    '''DBSCAN算法
    :param data(mat):需要聚类的数据集
    :param eps(float):半径
    :param MinPts(int):半径内最少的数据点数
    :return:
        types(mat):每个样本的类型：核心点、边界点、噪音点
        sub_class(mat):每个样本所属的类别
    '''
    m = np.shape(data)[0]
    # 在types中，1为核心点，0为边界点，-1为噪音点
    types = np.mat(np.zeros((1, m)))
    sub_class = np.mat(np.zeros((1, m)))
    # 用于判断该点是否处理过，0表示未处理过
    dealt = np.mat(np.zeros((m, 1)))
    # 计算每个数据点之间的距离
    dis = distance(data)
    # 用于标记类别
    number = 1

    # 对每一个点进行处理
    for i in range(m):
        # 找到未处理的点
        if dealt[i, 0] == 0:
            # 找到第i个点到其他所有点的距离
            D = dis[i,]
            # 找到半径eps内的所有点
            ind = find_eps(D, eps)
            # 区分点的类型
            # 边界点
            if len(ind) > 1 and len(ind) < MinPts + 1:
                types[0, i] = 0
                sub_class[0, i] = 0
            # 噪音点
            if len(ind) == 1:
                types[0, i] = -1
                sub_class[0, i] = -1
                dealt[i, 0] = 1
            # 核心点
            if len(ind) >= MinPts + 1:
                types[0, i] = 1
                for x in ind:
                    sub_class[0, x] = number
                # 判断核心点是否密度可达
                while len(ind) > 0:
                    dealt[ind[0], 0] = 1
                    D = dis[ind[0],]
                    tmp = ind[0]
                    del ind[0]
                    ind_1 = find_eps(D, eps)

                    if len(ind_1) > 1:  # 处理非噪音点
                        for x1 in ind_1:
                            sub_class[0, x1] = number
                        if len(ind_1) >= MinPts + 1:
                            types[0, tmp] = 1
                        else:
                            types[0, tmp] = 0

                        for j in range(len(ind_1)):
                            if dealt[ind_1[j], 0] == 0:
                                dealt[ind_1[j], 0] = 1
                                ind.append(ind_1[j])
                                sub_class[0, ind_1[j]] = number
                number += 1

    # 最后处理所有未分类的点为噪音点
    ind_2 = ((sub_class == 0).nonzero())[1]
    for x in ind_2:
        sub_class[0, x] = -1
        types[0, x] = -1

    return types, sub_class


def epsilon(data, MinPts):
    '''计算最佳半径
    input:  data(mat):训练数据
            MinPts(int):半径内的数据点的个数
    output: eps(float):半径
    '''
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps
