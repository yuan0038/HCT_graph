# -*- coding: utf-8 -*-
# @Time : 2023/4/14 3:55 PM
# @Author : Li Zhany
# @Email : 949777411@qq.com
# @File : visual_graph.py
# @Project : HCT_graph
import cv2
from matplotlib import pyplot as plt

def setAx(ax):
    ax.spines['bottom'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    ax.invert_yaxis()
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_position(('data', 0))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
def visual_graph(img,nodes,save_name):
    for i in range(nodes.shape[0]):
        y, x, Rx, Ry = int(nodes[i][0]), int(nodes[i][1]), int(nodes[i][2]), int(nodes[i][3])
        cv2.rectangle(img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    setAx(ax)
    plt.savefig(f"{save_name}.png",bbox_inches='tight' )