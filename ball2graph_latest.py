import argparse
import math
import os
import cv2
import numpy as np
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from collections import Counter
import networkx as nx
from collections import Counter
import scipy.sparse as sp
import itertools
import random
random.seed(0)

# 返回所有像素梯度信息
def get_grad(img, dir = None):
    """
    :param img:
    :return: map{(x,y):grad} and grad matric
    """
    # get grad list
    # 分别计算x、y方向：右减左，下减上
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # 梯度矩阵 : same shape with the img

    return sobelxy2

def softmax(x):
    """
    :param x: matric
    :return: softmax(matric)
    """
    exp = np.exp(x)
    sum = np.sum(exp, axis=0, keepdims=True)
    return exp / sum

def cal_bound(img, center, Rx, Ry):
    """
    :param img:  原图，需要获取原图的大小
    :param center: 中心点坐标
    :param Rx, Ry: 不同方向的半径
    :return: 上下左右边界
    """
    h, w = img.shape
    left = center[1] - Rx
    right = center[1] + Rx
    up = center[0] - Ry
    down = center[0] + Ry

    if left < 0 :
        left = 0
    if right >= w :
        right = w - 1
    if up < 0 :
        up = 0
    if down >= h :
        down = h - 1

    return int(left), int(right), int(up), int(down)

def center_select(img_grad, img_label):
    """
    :param img_grad:
    :param img_label:
    :return:
    """
    minPix_xy = np.where(img_grad == img_grad.min()) # 返回一个二维元组，前一个为所有梯度最小的点的横坐标，后一个为所有梯度最小的点的纵坐标
    random.seed(777)
    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)  # 随机挑选一个中心点
        #print(len(minPix_xy[0]))
        if (img_label[minPix_xy[0][pos],minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos],minPix_xy[1][pos]]

def cal_Radius_1(img, center, purity, threshold, var_threshold): # 半径计算1：半径初始化为0，纯度逐渐增加，适用于中心点梯度较小的情况
    """
    :param img: 输入图片
    :param center:  中心点坐标 [x, y]
    :param purity:  1 - （异类点个数 / 总个数)
    :param threshold:  判断是否是异类点， 与中心点灰度值的差值的绝对值 / 中心点的灰度值
    :return: Rx, Ry, 输入中心点对应的半径
    """
    Rx = 0 # 初始化半径
    Ry = 0

    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0

    center_value = int(img[center[0], center[1]])

    while True:

        if flag_x == True and flag_y == True:
            item_count += 1
        else:
            if flag_x:
                item_count = 1
            if flag_y:
                item_count = 2

        # print(item_count)

        if flag_x and item_count % 2 != 0:
            Rx += 1

        if flag_y and item_count % 2 == 0:
            Ry += 1

        #计算切片边界
        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)  # 当前总像素点

        if pixNum == temp_pixNum:
            return Rx, Ry

        # print("当前中心为:", [center[0], center[1]], "当前半径为:", Rx, Ry ,"总像素点数为:", pixNum)

        # 计算异类点个数
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])

        # print("异类点个数为：", count)

        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])
        temp_pixNum = pixNum

        # print("当前纯度为：", temp_purity)
        if temp_purity > purity and var < var_threshold:
            if purity < 0.99:
                purity = purity * 1.005
            else:
                purity = 0.99
            flag = True
        else:
            flag = False

        if flag == False and item_count % 2 != 0:
            flag_x = False
            Rx -= 1

        if flag == False and item_count % 2 == 0:
            flag_y = False
            Ry -= 1

        # print(flag_x, flag_y)

        if flag_x == False and flag_y == False:
            return Rx, Ry

def cal_Radius_2(img, center, purity, threshold, var_threshold): # 半径计算2：半径初始化为1，纯度固定不变（默认0.5），方差设置较大，适用于中心点梯度较大的情况
    """
    :param img: 输入图片
    :param center:  中心点坐标 [x, y]
    :param purity:  1 - （异类点个数 / 总个数)
    :param threshold:  判断是否是异类点， 与中心点灰度值的差值的绝对值 / 中心点的灰度值
    :return: Rx, Ry, 输入中心点对应的半径
    """
    Rx = 1 # 初始化半径
    Ry = 1

    flag = True
    flag_x = True
    flag_y = True
    item_count = 0
    temp_pixNum = 0

    center_value = int(img[center[0], center[1]])

    while True:

        if flag_x == True and flag_y == True:
            item_count += 1
        else:
            if flag_x:
                item_count = 1
            if flag_y:
                item_count = 2

        # print(item_count)

        if flag_x and item_count % 2 != 0:
            Rx += 1

        if flag_y and item_count % 2 == 0:
            Ry += 1

        #计算切片边界
        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)  # 当前总像素点

        if pixNum == temp_pixNum:
            return Rx, Ry

        # print("当前中心为:", [center[0], center[1]], "当前半径为:", Rx, Ry ,"总像素点数为:", pixNum)

        # 计算异类点个数
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])


        # print("异类点个数为：", count)

        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])

        temp_pixNum = pixNum

        # print("当前纯度为：", temp_purity)
        if temp_purity > purity and var < var_threshold:
            # if purity < 0.99:
            #     purity += 0.01
            # else:
            #     purity = 0.99
            flag = True
        else:
            flag = False

        if flag == False and item_count % 2 != 0:
            flag_x = False
            Rx -= 1

        if flag == False and item_count % 2 == 0:
            flag_y = False
            Ry -= 1

        # print(flag_x, flag_y)

        if flag_x == False and flag_y == False:
            return Rx, Ry

def calulate_weight(img, center_1, center_2):
    """
    :param img:
    :param center_1:
    :param center_2:
    :return: 两个粒矩相交像素点个数
    """
    left_1, right_1, up_1, down_1 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    left_2, right_2, up_2, down_2 = cal_bound(img, center_2[0], center_2[1], center_2[2])


    x_list = [up_1, up_2, down_1, down_2]
    y_list = [left_1, left_2, right_1, right_2]
    x_list.sort()
    y_list.sort()

    res = (x_list[2] - x_list[1] + 1) * (y_list[2] - y_list[1] + 1)

    return res

def calulate_A_and_B(img, center_1, center_2):
    """
    :param center_1:
    :param center_2:
    :return: 两个粒矩的像素点数和
    """
    x1, x2, y1, y2 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    x3, x4, y3, y4 = cal_bound(img, center_2[0], center_2[1], center_2[2])

    return (x2 - x1 + 1) * (y2 - y1 + 1) + (x4 - x3 + 1) * (y4 - y3 + 1)


def ball2graph(img, purity_1=0.9, threshold=10, var_threshold_1=20):  # 论文里 cifar-10 用的这个聚类代码

    RGB_img = np.transpose(img, (2, 0, 1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()  # 计算梯度图最大值
    # print(grad_median)
    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    start = time.time()
    while 0 in img_label: # 存在没有被划分的点
        temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点

        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        #if img_grad[temp_center[0]][temp_center[1]] <= grad_median:
        Rx, Ry = cal_Radius_1(img, temp_center, purity_1, threshold, var_threshold_1)
        #else:
            #Rx, Ry = cal_Radius_2(img, temp_center, purity_2, threshold, var_threshold_2)

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)  # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作

        # 添加粒矩 （[x, y], Rx, Ry） 一个中心一个元组.最基础特征:中心点坐标， Rx， Ry
        # center.append((temp_center, Rx, Ry))

        # 添加粒矩  RGB图特征 -> 中心点坐标， Rx， Ry, 三个通道的均值和方差, 中心点梯度值
        center.append((
                temp_center,
                Rx,Ry,
                RGB_img[0][up:down + 1, left:right + 1].mean(), np.var(RGB_img[0][up:down + 1, left:right + 1]),
                RGB_img[1][up:down + 1, left:right + 1].mean(), np.var(RGB_img[1][up:down + 1, left:right + 1]),
                RGB_img[2][up:down + 1, left:right + 1].mean(), np.var(RGB_img[2][up:down + 1, left:right + 1]),
                img_grad[temp_center[0], temp_center[1]])) # img[up:down + 1, left:right + 1].max

        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1

    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print(center_count)
    # 聚类完成，开始构建 Graph
    # 初始化图
    g = nx.Graph()

    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))

    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i+1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            # 两个中心点的 x,y 距离分别和 两个粒矩的 Rx,Ry 之和进行比较，判断两个粒矩之间的位置关系（相交，相接，相离）
            # if (abs(center_1[0][0] - center_2[0][0]) - 1) <= center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) <= center_1[1] + center_2[1]:  # 相接有边
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                g.add_edge(str(i), str(j))

    # 3. 生成 GNN 需要的数据
    a = nx.to_numpy_matrix(g)

    # 每个节点的度、特征向量中心性、中介中心性、接近中心性
    other_f = np.zeros((len(center), 2))
    # eigen_dict = nx.eigenvector_centrality(g, max_iter=10000)  # 特征向量中心性
    # betw_dict = nx.betweenness_centrality(g)  # 中介中心性
    # close_dict = nx.closeness_centrality(g)  # 接近中心性
    degree_dict = nx.degree(g)  # 节点的度数
    cluster_dict = nx.clustering(g)  # 节点的聚类系数
    # pagerank_dict = nx.pagerank(g, max_iter=10000)  # 节点的平稳概率值
    for i in range(len(center)):
        # other_f[i] = [eigen_dict[str(i)], betw_dict[str(i)], close_dict[str(i)], degree_dict[str(i)], cluster_dict[str(i)], pagerank_dict[str(i)]]
        other_f[i] = [degree_dict[str(i)], cluster_dict[str(i)]]

    # 邻接矩阵 adj
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_array = np.zeros((len(center), 11))  # 节点属性 (需要在粒矩生成的时候将属性添加进去才能操作)
    edge_attr = np.zeros((len(adj[0]), 3))  # 边特征
    center_ = np.zeros((len(center), 4))  # 粒矩基础属性 -> 中心坐标, Rx, Ry

    # 边特征 edge_attr -> ①两个粒矩相交的像素点个数，②两个粒矩相交的像素点占两个粒矩总像素点的个数，③两个粒矩中心的距离
    for i in range(len(adj[0])):
        temp = calulate_weight(img, center[adj[0][i]], center[adj[1][i]])
        temp_iou = calulate_A_and_B(img, center[adj[0][i]], center[adj[1][i]])
        center_dis = math.sqrt((center[adj[0][i]][0][0] - center[adj[1][i]][0][0]) ** 2 + (center[adj[0][i]][0][1] - center[adj[1][i]][0][1]) ** 2)
        edge_attr[i] = [temp, temp / temp_iou, center_dis]
    # 粒矩相交像素点个数和粒矩中心距离归一化
    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()
    edge_attr[:, 2] = edge_attr[:, 2] / edge_attr[:, 2].max()

    # 生成节点属性和粒矩基础属性数组
    for id in range(len(center)):
        # 节点属性 center_array
        center_array[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2], center[id][3], center[id][4],
                            center[id][5], center[id][6], center[id][7], center[id][8], center[id][9]]
        # 粒矩基础属性 center_
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

    center_array = np.hstack((center_array, other_f))

    return center_array, adj, edge_attr, center_

def ball2graph_CNN_GCN(img, purity_1=0.9, threshold=10, var_threshold_1=20): # 带有掩码矩阵的聚类

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()  # 计算梯度图最大值

    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    start = time.time()
    while 0 in img_label: # 存在没有被划分的点
        temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点
        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        Rx, Ry = cal_Radius_1(img, temp_center, purity_1, threshold, var_threshold_1)
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)

        # 生成该节点的 mask
        temp_mask = np.full((h, w), False, dtype=bool) # 全 False
        temp_mask[up:down + 1, left:right + 1] = True

        # 计算粒矩像素点个数
        temp_pix_num = (right - left + 1) * (down - up + 1)
        center.append((temp_center, Rx, Ry, temp_mask, temp_pix_num))

        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1

    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print(center_count)

    # 聚类完成，开始构建 Graph
    # 初始化图
    g = nx.Graph()

    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))

    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i+1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            # 两个中心点的 x,y 距离分别和 两个粒矩的 Rx,Ry 之和进行比较，判断两个粒矩之间的位置关系（相交，相接，相离）
            # if (abs(center_1[0][0] - center_2[0][0]) - 1) <= center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) <= center_1[1] + center_2[1]:  # 相接有边
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                g.add_edge(str(i), str(j))

    # 3. 生成 GNN 需要的数据
    a = nx.to_numpy_matrix(g)

    # 邻接矩阵 adj
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_ = np.zeros((len(center), 4), dtype=int)  # 粒矩基础属性 -> 中心坐标, Rx, Ry
    pix_num_list = np.zeros(len(center), dtype=int)
    mask = np.full((len(center), h, w), False, dtype=bool)

    # 生成粒矩基础节点数组和掩码矩阵
    for id in range(len(center)):
        # 节点属性 center_array
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

        # 粒矩掩码 mask
        mask[id] = center[id][3]

        # 粒矩像素点个数数组
        pix_num_list[id] = center[id][4]

    return center_, adj, mask, pix_num_list


def ball2graph_random_select_center(img, purity=0.9, threshold=10, var_threshold=20): # 随机选取中心点
    
    RGB_img = np.transpose(img, (2, 0, 1))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 读取灰度图
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    # print(grad_median)

    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    temp_center_list_x = random.sample(range(0, h), h)
    temp_center_list_y = random.sample(range(0, w), w)
    temp_center_list = [] # 所有像素点的坐标
    for i in temp_center_list_x:
        for j in temp_center_list_y:
            temp_center_list.append([i ,j])
    random.shuffle(temp_center_list) # 打乱所有像素点的坐标

    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    start = time.time()
    while 0 in img_label: # 存在没有被划分的点
        # temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点
        # 随机选中心点
        temp_center = temp_center_list[random.randint(0, h * w - 1)] # 随机找一个点
        while img_label[temp_center[0], temp_center[1]]: # 如果这个点被划分过，就再随机找下一个
            temp_center = temp_center_list[random.randint(0, h * w - 1)]

        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        # if img_grad[temp_center[0]][temp_center[1]] <= grad_median:
        Rx, Ry = cal_Radius_1(img, temp_center, purity, threshold, var_threshold)
        # else:
        #     Rx, Ry = cal_Radius_2(img, temp_center, purity_2, threshold, var_threshold_2)
        if Rx <= 1 and Ry <= 1:
            Rx = 1
            Ry = 1

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)  # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作

        # 添加粒矩 （[x, y], Rx, Ry） 一个中心一个元组.最基础特征:中心点坐标， Rx， Ry
        # center.append((temp_center, Rx, Ry))

        # 添加粒矩  RGB图特征 -> 中心点坐标， Rx， Ry, 三个通道的均值和方差, 中心点梯度值
        center.append((
                temp_center,
                Rx,Ry,
                RGB_img[0][up:down + 1, left:right + 1].mean(), np.var(RGB_img[0][up:down + 1, left:right + 1]),
                RGB_img[1][up:down + 1, left:right + 1].mean(), np.var(RGB_img[1][up:down + 1, left:right + 1]),
                RGB_img[2][up:down + 1, left:right + 1].mean(), np.var(RGB_img[2][up:down + 1, left:right + 1]),
                img_grad[temp_center[0], temp_center[1]]))

        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        # img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1

    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print(center_count)
    # 聚类完成，开始构建 Graph
   # 初始化图
    g = nx.Graph()

    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))

    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i+1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            # 两个中心点的 x,y 距离分别和 两个粒矩的 Rx,Ry 之和进行比较，判断两个粒矩之间的位置关系（相交，相接，相离）
            # if (abs(center_1[0][0] - center_2[0][0]) - 1) <= center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) <= center_1[1] + center_2[1]:  # 相接有边
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                g.add_edge(str(i), str(j))

    # 3. 生成 GNN 需要的数据
    a = nx.to_numpy_matrix(g)

    # 每个节点的度、特征向量中心性、中介中心性、接近中心性
    other_f = np.zeros((len(center), 2))
    # eigen_dict = nx.eigenvector_centrality(g, max_iter=10000)  # 特征向量中心性
    # betw_dict = nx.betweenness_centrality(g)  # 中介中心性
    # close_dict = nx.closeness_centrality(g)  # 接近中心性
    degree_dict = nx.degree(g)  # 节点的度数
    cluster_dict = nx.clustering(g)  # 节点的聚类系数
    # pagerank_dict = nx.pagerank(g, max_iter=10000)  # 节点的平稳概率值
    for i in range(len(center)):
        # other_f[i] = [eigen_dict[str(i)], betw_dict[str(i)], close_dict[str(i)], degree_dict[str(i)], cluster_dict[str(i)], pagerank_dict[str(i)]]
        other_f[i] = [degree_dict[str(i)], cluster_dict[str(i)]]

    # 邻接矩阵 adj
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_array = np.zeros((len(center), 11))  # 节点属性 (需要在粒矩生成的时候将属性添加进去才能操作)
    edge_attr = np.zeros((len(adj[0]), 3))  # 边特征
    center_ = np.zeros((len(center), 4))  # 粒矩基础属性 -> 中心坐标, Rx, Ry

    # 边特征 edge_attr -> ①两个粒矩相交的像素点个数，②两个粒矩相交的像素点占两个粒矩总像素点的个数，③两个粒矩中心的距离
    for i in range(len(adj[0])):
        temp = calulate_weight(img, center[adj[0][i]], center[adj[1][i]])
        temp_iou = calulate_A_and_B(img, center[adj[0][i]], center[adj[1][i]])
        center_dis = math.sqrt((center[adj[0][i]][0][0] - center[adj[1][i]][0][0]) ** 2 + (center[adj[0][i]][0][1] - center[adj[1][i]][0][1]) ** 2)
        edge_attr[i] = [temp, temp / temp_iou, center_dis]
    # 粒矩相交像素点个数和粒矩中心距离归一化
    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()
    edge_attr[:, 2] = edge_attr[:, 2] / edge_attr[:, 2].max()

    # 生成节点属性和粒矩基础属性数组
    for id in range(len(center)):
        # 节点属性 center_array
        center_array[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2], center[id][3], center[id][4],
                            center[id][5], center[id][6], center[id][7], center[id][8], center[id][9]]
        # 粒矩基础属性 center_
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

    center_array = np.hstack((center_array, other_f))

    return center_array, adj, edge_attr, center_


if __name__ == '__main__':

    #start = time.time()
    img_name = "/home/ubuntu/lxd-workplace/zh/imagenet/train/bird/n01608432_37.JPEG"  # 图片路径

    RGB_img = cv2.imread(img_name)  # 读取RGB图用来进行特征提取
    RGB_img = np.transpose(RGB_img, (2, 0, 1))
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    grad_median = np.median(img_label)  # 计算梯度图中位数
    max_Grad = img_grad.max()  # 计算梯度图最大值
    # print(grad_median)

    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    temp_center_list_x = random.sample(range(0, h), h)
    temp_center_list_y = random.sample(range(0, w), w)
    temp_center_list = [] # 所有像素点的坐标
    for i in temp_center_list_x:
        for j in temp_center_list_y:
            temp_center_list.append([i ,j])
    random.shuffle(temp_center_list) # 打乱所有像素点的坐标

    center = []  # 创建中心列表
    center_count = 0  # 中心点个数计数

    # 参数设置
    purity = 0.7
    threshold = 30
    var_threshold = 30

    start = time.time()
    while 0 in img_label: # 存在没有被划分的点
        # temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点
        # 随机选中心点
        temp_center = temp_center_list[random.randint(0, h * w - 1)] # 随机找一个点
        while img_label[temp_center[0], temp_center[1]]: # 如果这个点被划分过，就再随机找下一个
            temp_center = temp_center_list[random.randint(0, h * w - 1)]

        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        # if img_grad[temp_center[0]][temp_center[1]] <= grad_median:
        Rx, Ry = cal_Radius_1(img, temp_center, purity, threshold, var_threshold)
        # else:
        #     Rx, Ry = cal_Radius_2(img, temp_center, purity_2, threshold, var_threshold_2)
        if Rx <= 1 and Ry <= 1:
            Rx = 1
            Ry = 1

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)  # 计算实际的矩形在图片中的位置(存在粒矩大小超出图像范围，所以不能直接用半径进行切片)，方便后续使用切片进行特征提取等操作

        # 添加粒矩 （[x, y], Rx, Ry） 一个中心一个元组.最基础特征:中心点坐标， Rx， Ry
        # center.append((temp_center, Rx, Ry))

        # 添加粒矩  RGB图特征 -> 中心点坐标， Rx， Ry, 三个通道的均值和方差, 中心点梯度值
        center.append((
                temp_center,
                Rx,Ry,
                RGB_img[0][up:down + 1, left:right + 1].mean(), np.var(RGB_img[0][up:down + 1, left:right + 1]),
                RGB_img[1][up:down + 1, left:right + 1].mean(), np.var(RGB_img[1][up:down + 1, left:right + 1]),
                RGB_img[2][up:down + 1, left:right + 1].mean(), np.var(RGB_img[2][up:down + 1, left:right + 1]),
                img_grad[temp_center[0], temp_center[1]]))

        # 将本次迭代生成的粒矩包含的像素点标记 （下次迭代就不会选取这些点作为中心点）
        img_label[up:down + 1, left:right + 1] = 1
        # 将本次迭代生成的粒矩包含的像素点对应位置的梯度设为梯度最大值
        # img_grad[up:down + 1, left:right + 1] = max_Grad
        # 粒矩计数
        center_count += 1

    end = time.time()
    print("粒矩聚类时间:%.2f秒" % (end - start))
    print(center_count)
    # 聚类完成，开始构建 Graph
   # 初始化图
    g = nx.Graph()

    # 1. 添加节点
    for i in range(len(center)):
        g.add_node(str(i))

    # 2. 生成边 (使用粒矩的位置关系进行边的生成)
    for i in range(len(center)):
        for j in range(i+1, len(center)):
            center_1 = center[i]
            center_2 = center[j]
            # 两个中心点的 x,y 距离分别和 两个粒矩的 Rx,Ry 之和进行比较，判断两个粒矩之间的位置关系（相交，相接，相离）
            # if (abs(center_1[0][0] - center_2[0][0]) - 1) <= center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) <= center_1[1] + center_2[1]:  # 相接有边
            if (abs(center_1[0][0] - center_2[0][0]) - 1) < center_1[2] + center_2[2] and (abs(center_1[0][1] - center_2[0][1]) - 1) < center_1[1] + center_2[1]:  # 相接无边
                g.add_edge(str(i), str(j))

    # 3. 生成 GNN 需要的数据
    a = nx.to_numpy_matrix(g)

    # 每个节点的度、特征向量中心性、中介中心性、接近中心性
    other_f = np.zeros((len(center), 2))
    # eigen_dict = nx.eigenvector_centrality(g, max_iter=10000)  # 特征向量中心性
    # betw_dict = nx.betweenness_centrality(g)  # 中介中心性
    # close_dict = nx.closeness_centrality(g)  # 接近中心性
    degree_dict = nx.degree(g)  # 节点的度数
    cluster_dict = nx.clustering(g)  # 节点的聚类系数
    # pagerank_dict = nx.pagerank(g, max_iter=10000)  # 节点的平稳概率值
    for i in range(len(center)):
        # other_f[i] = [eigen_dict[str(i)], betw_dict[str(i)], close_dict[str(i)], degree_dict[str(i)], cluster_dict[str(i)], pagerank_dict[str(i)]]
        other_f[i] = [degree_dict[str(i)], cluster_dict[str(i)]]

    # 邻接矩阵 adj
    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))

    center_array = np.zeros((len(center), 11))  # 节点属性 (需要在粒矩生成的时候将属性添加进去才能操作)
    edge_attr = np.zeros((len(adj[0]), 3))  # 边特征
    center_ = np.zeros((len(center), 4))  # 粒矩基础属性 -> 中心坐标, Rx, Ry

    # 边特征 edge_attr -> ①两个粒矩相交的像素点个数，②两个粒矩相交的像素点占两个粒矩总像素点的个数，③两个粒矩中心的距离
    for i in range(len(adj[0])):
        temp = calulate_weight(img, center[adj[0][i]], center[adj[1][i]])
        temp_iou = calulate_A_and_B(img, center[adj[0][i]], center[adj[1][i]])
        center_dis = math.sqrt((center[adj[0][i]][0][0] - center[adj[1][i]][0][0]) ** 2 + (center[adj[0][i]][0][1] - center[adj[1][i]][0][1]) ** 2)
        edge_attr[i] = [temp, temp / temp_iou, center_dis]
    # 粒矩相交像素点个数和粒矩中心距离归一化
    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()
    edge_attr[:, 2] = edge_attr[:, 2] / edge_attr[:, 2].max()

    # 生成节点属性和粒矩基础属性数组
    for id in range(len(center)):
        # 节点属性 center_array
        center_array[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2], center[id][3], center[id][4],
                            center[id][5], center[id][6], center[id][7], center[id][8], center[id][9]]
        # 粒矩基础属性 center_
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

    center_array = np.hstack((center_array, other_f))