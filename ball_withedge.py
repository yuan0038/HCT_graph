import argparse
import math
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import time
import itertools
from collections import Counter
import networkx as nx
import numpy.ma as npm
import random
import scipy.sparse as sp


# import torch
def get_grad(img, dir=None):
    """
    :param img:
    :return: map{(x,y):grad} and grad matric
    """
    # get grad list
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

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

    if left < 0:
        left = 0
    if right >= w:
        right = w - 1
    if up < 0:
        up = 0
    if down >= h:
        down = h - 1

    return left, right, up, down


def center_select(img_grad, img_label):
    """
    :param img_grad:
    :param img_label:
    :return:
    """
    minPix_xy = np.where(img_grad == img_grad.min())
    random.seed(3)
    while True:

        pos = random.randint(0, len(minPix_xy[0]) - 1)
        # print(len(minPix_xy[0]))
        if (img_label[minPix_xy[0][pos], minPix_xy[1][pos]] != 1):
            return [minPix_xy[0][pos], minPix_xy[1][pos]]


def cal_Radius(img, center, purity, threshold, var_threshold):
    """
    :param img: 输入图片
    :param center:  中心点坐标 [x, y]
    :param purity:  1 - （异类点个数 / 总个数)
    :param threshold:  判断是否是异类点， 与中心点灰度值的差值的绝对值 / 中心点的灰度值
    :return: Rx, Ry, 输入中心点对应的半径

    1. 初始化 Rx, Ry 为 1
    2. 向外扩策略： ① 两个方向交替向外扩展；

    不要单像素做法 （先不管）：假设 Rx = 1 的时候不满足纯度要求， 先把它置为 0， 观察 Ry = 1 是否满足纯度要求， 如果也不满足， 返回 Rx, Ry 都为 1 ； 如果满足， Rx,

    可以存在单像素 ： 直接交替增加，

    """
    Rx = 0  # 初始化半径
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

        if flag_y and item_count % 2 != 0:
            Ry += 1

        # 计算切片边界
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


def calulate_weight(img, center_1, center_2):
    """
    :param img:
    :param center_1:
    :param center_2:
    :return:
    """
    h, w = img.shape

    x_list = [center_1[0][0] - center_1[1], center_1[0][0] + center_1[1], center_2[0][0] - center_2[1],
              center_2[0][0] + center_2[1]]
    for i in range(len(x_list)):
        if x_list[i] > h:
            x_list[i] = h - 1
        if x_list[i] < 0:
            x_list[i] = 0
    x_list.sort()

    y_list = [center_1[0][1] - center_1[2], center_1[0][1] + center_1[2], center_2[0][1] - center_2[2],
              center_2[0][1] + center_2[2]]
    for i in range(len(y_list)):
        if y_list[i] > w:
            y_list[i] = w - 1
        if y_list[i] < 0:
            y_list[i] = 0
    y_list.sort()

    res = (x_list[2] - x_list[1] + 1) * (y_list[2] - y_list[1] + 1)

    return res


def calulate_A_and_B(img, center_1, center_2):
    """
    :param center_1:
    :param center_2:
    :return:
    """
    x1, x2, y1, y2 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    x3, x4, y3, y4 = cal_bound(img, center_2[0], center_2[1], center_2[2])

    return (x2 - x1 + 1) * (y2 - y1 + 1) + (x4 - x3 + 1) * (y4 - y3 + 1)


def ball2graph(img, purity=0.7, threshold=20, var_threshold=30000):
    RGB_img = np.transpose(img, (2, 0, 1))
    # print(RGB_img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图

    max_Grad = img_grad.max()
    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    pixLabel_list = []
    pixWeight_list = []
    for i in range(h):
        pixLabel_list.append([])
        for j in range(w):
            pixLabel_list[i].append([])

    center = []  # 创建中心列表
    center_count = 0

    start = time.time()
    # mask = np.zeros((h, w), dtype=bool)

    while 0 in img_label:  # 存在没有被划分的点
        # for i in range(100):
        # print("第%d次迭代：", format(center_count + 1))
        temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点
        # print("当前中心点为：", temp_center)

        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        Rx, Ry = cal_Radius(img, temp_center, purity, threshold, var_threshold)

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)

        img_label[up:down + 1, left:right + 1] = 1

        # img_grad = npm.array(img_grad, mask = mask)

        # 给像素点打标 (看列表的列表能不能用切片的方法赋值？)
        for i in range(up, down + 1):
            for j in range(left, right + 1):
                pixLabel_list[i][j].append(center_count)

        # 添加中心 （[x, y], Rx, Ry, 均值， 总体方差, 梯度） 一个中心一个元组
        # center.append(
        #     (temp_center, Rx, Ry, img[up:down + 1, left:right + 1].mean(), np.var(img[up:down + 1, left:right + 1]),
        #      img_grad[temp_center[0], temp_center[1]]))

        # 添加中心 （[x, y], Rx，Ry, R均值，G均值，B均值，R总体方差，G总体方差，B总体方差, 灰度图梯度）
        center.append(
            (
                temp_center,
                Rx,Ry,
                RGB_img[0][up:down + 1, left:right + 1].mean(), np.var(RGB_img[0][up:down + 1, left:right + 1]),
                RGB_img[1][up:down + 1, left:right + 1].mean(), np.var(RGB_img[1][up:down + 1, left:right + 1]),
                RGB_img[2][up:down + 1, left:right + 1].mean(), np.var(RGB_img[2][up:down + 1, left:right + 1]),
                img_grad[temp_center[0], temp_center[1]],
                # img[up:down + 1, left:right + 1].max(), img[up:down + 1, left:right + 1].min()
            )
        )
        img_grad[up:down + 1, left:right + 1] = max_Grad
        center_count += 1

    end = time.time()
    print("循环运行时间:%.2f秒" % (end - start))

    # num_count = np.zeros((len(center), len(center)))

    g = nx.Graph()

    for i in range(len(center)):
        g.add_node(str(i))

    pos_dict = {}
    for i in range(len(center)):
        pos_dict[str(i)] = [center[i][0][0], center[i][0][1]]
        # print([center[i][0][0], center[i][0][1]])

    for i in pixLabel_list:
        for j in i:
            # print(j)
            if len(j) > 1:
                temp_point_list = list(itertools.permutations(j, 2))
                # print(temp_point_list)

                for temp in temp_point_list:
                    # print(temp)
                    # num_count[temp[0]][temp[1]] += 1
                    g.add_edge(str(temp[0]), str(temp[1]))

    a = nx.to_numpy_matrix(g)

    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))
    # adj = torch.LongTensor(indices)

    node_id = list(g.nodes())

    center_array = np.zeros((len(center), 11))
    edge_attr = np.zeros((len(adj[0]), 2))
    center_index = np.zeros((len(center), 2))
    # node_dis = np.zeros((adj.shape[0], adj.shape[1]))

    for i in range(len(adj[0])):
        temp = calulate_weight(img, center[adj[0][i]], center[adj[1][i]])
        temp_iou = calulate_A_and_B(img, center[adj[0][i]], center[adj[1][i]])
        edge_attr[i] = [temp, temp / temp_iou]

    edge_attr[:, 0] = edge_attr[:, 0] / edge_attr[:, 0].max()

    for id in range(len(node_id)):
        # center_array[id] = [center[int(node_id[id])][2], center[int(node_id[id])][3], center[int(node_id[id])][4],
        #                      center[int(node_id[id])][5], center[int(node_id[id])][6]]

        # center_array[id] = [center[int(node_id[id])][0][0], center[int(node_id[id])][0][1],
        #                     center[int(node_id[id])][1] + center[int(node_id[id])][2], center[int(node_id[id])][3],
        #                     center[int(node_id[id])][4], center[int(node_id[id])][5]]

        center_array[id] = [center[int(node_id[id])][0][0], center[int(node_id[id])][0][1],
                            center[int(node_id[id])][1],center[int(node_id[id])][2],
                            center[int(node_id[id])][3], center[int(node_id[id])][4],
                            center[int(node_id[id])][5], center[int(node_id[id])][6],
                            center[int(node_id[id])][7], center[int(node_id[id])][8],
                            center[int(node_id[id])][9]]

        # center_array[id] = np.random.normal(center[int(node_id[id])][2], center[int(node_id[id])][3], 1)
        center_index[id] = center[int(node_id[id])][0]
    # print(center_index)
    return center_array, adj, center_index, edge_attr


if __name__ == '__main__':
    img = cv2.imread("./data/ILSVRC2012_val_00027391.JPEG", 1)  # 读取灰度图
    # center_array, adj, center_index, edge = ball2graph(img, purity=0.7, threshold=10, var_threshold=20)

    purity = 0.7
    threshold = 10
    var_threshold = 20

    RGB_img = np.transpose(img, (2, 0, 1))
    # print(RGB_img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯滤波
    img_label = np.zeros(img.shape)  # 创建label矩阵
    img_grad = get_grad(img)  # 计算梯度图
    max_Grad = img_grad.max()
    h, w = img.shape  # 输入图片的 高 h -> x 和宽 w -> y
    pixLabel_list = []
    pixWeight_list = []
    for i in range(h):
        pixLabel_list.append([])
        for j in range(w):
            pixLabel_list[i].append([])

    center = []  # 创建中心列表
    center_count = 0

    start = time.time()
    # mask = np.zeros((h, w), dtype=bool)

    while 0 in img_label:  # 存在没有被划分的点
        # for i in range(100):
        # print("第%d次迭代：", format(center_count + 1))
        temp_center = center_select(img_grad, img_label)  # 选择一个梯度最小且没有被划分过的点为中心点
        # print("当前中心点为：", temp_center)

        # 计算半径 Rx, Ry -> x 方向的半径与 y 方向的半径
        Rx, Ry = cal_Radius(img, temp_center, purity, threshold, var_threshold)

        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)

        img_label[up:down + 1, left:right + 1] = 1
        img_grad[up:down + 1, left:right + 1] = max_Grad
        # img_grad = npm.array(img_grad, mask = mask)

        # 给像素点打标 (看列表的列表能不能用切片的方法赋值？)
        for i in range(up, down + 1):
            for j in range(left, right + 1):
                pixLabel_list[i][j].append(center_count)

        # 添加中心 （[x, y], Rx, Ry, 均值， 总体方差, 梯度） 一个中心一个元组
        # center.append(
        #     (temp_center, Rx, Ry, img[up:down + 1, left:right + 1].mean(), np.var(img[up:down + 1, left:right + 1]),
        #      img_grad[temp_center[0], temp_center[1]]))

        # 添加中心 （[x, y], Rx，Ry, R均值，G均值，B均值，R总体方差，G总体方差，B总体方差, 灰度图梯度）
        center.append(
            (
                temp_center,
                Rx, Ry,
                RGB_img[0][up:down + 1, left:right + 1].mean(), np.var(RGB_img[0][up:down + 1, left:right + 1]),
                RGB_img[1][up:down + 1, left:right + 1].mean(), np.var(RGB_img[1][up:down + 1, left:right + 1]),
                RGB_img[2][up:down + 1, left:right + 1].mean(), np.var(RGB_img[2][up:down + 1, left:right + 1]),
                img_grad[temp_center[0], temp_center[1]],
                # img[up:down + 1, left:right + 1].max(), img[up:down + 1, left:right + 1].min()
            )
        )

        center_count += 1

    end = time.time()
    print("循环运行时间:%.2f秒" % (end - start))

    # num_count = np.zeros((len(center), len(center)))

    g = nx.Graph()

    for i in range(len(center)):
        g.add_node(str(i))

    pos_dict = {}
    for i in range(len(center)):
        pos_dict[str(i)] = [center[i][0][0], center[i][0][1]]
        # print([center[i][0][0], center[i][0][1]])

    for i in pixLabel_list:
        for j in i:
            # print(j)
            if len(j) > 1:
                temp_point_list = list(itertools.permutations(j, 2))
                # print(temp_point_list)

                for temp in temp_point_list:
                    # print(temp)
                    # num_count[temp[0]][temp[1]] += 1
                    g.add_edge(str(temp[0]), str(temp[1]))

    a = nx.to_numpy_matrix(g)

    adj = a.A
    adj = sp.coo_matrix(adj)
    adj = np.vstack((adj.row, adj.col))
    # adj = torch.LongTensor(indices)

    node_id = list(g.nodes())

    center_array = np.zeros((len(center), 11))
    edge_attr = np.zeros((len(adj[0]), 2))
    center_index = np.zeros((len(center), 2))
    # node_dis = np.zeros((adj.shape[0], adj.shape[1]))

    for i in range(len(adj[0])):
        temp = calulate_weight(img, center[adj[0][i]], center[adj[1][i]])
        temp_iou = calulate_A_and_B(img, center[adj[0][i]], center[adj[1][i]])
        edge_attr[i] = [temp, temp / temp_iou]

    for id in range(len(node_id)):
        # center_array[id] = [center[int(node_id[id])][2], center[int(node_id[id])][3], center[int(node_id[id])][4],
        #                      center[int(node_id[id])][5], center[int(node_id[id])][6]]

        # center_array[id] = [center[int(node_id[id])][0][0], center[int(node_id[id])][0][1],
        #                     center[int(node_id[id])][1] + center[int(node_id[id])][2], center[int(node_id[id])][3],
        #                     center[int(node_id[id])][4], center[int(node_id[id])][5]]

        center_array[id] = [center[int(node_id[id])][0][0], center[int(node_id[id])][0][1],
                            center[int(node_id[id])][1], center[int(node_id[id])][2],
                            center[int(node_id[id])][3], center[int(node_id[id])][4],
                            center[int(node_id[id])][5], center[int(node_id[id])][6],
                            center[int(node_id[id])][7], center[int(node_id[id])][8],
                            center[int(node_id[id])][9]]

        # center_array[id] = np.random.normal(center[int(node_id[id])][2], center[int(node_id[id])][3], 1)
        center_index[id] = center[int(node_id[id])][0]
    # print(center_index)

