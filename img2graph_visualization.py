import matplotlib.pyplot as plt

from img2graph import *
random.seed(0)

if __name__ == '__main__':
    img_name = "./00010200.png"  # 图片路径
    RGB_img = cv2.imread(img_name)
    RGB_img = cv2.resize(RGB_img,(224,224))
    h, w = RGB_img.shape[0:2]
    center, g = img2graph(RGB_img)

    # 粒矩在原图片上的可视化
    for v in center:
        y, x, Rx, Ry = v[0][0], v[0][1], v[1], v[2]
        cv2.rectangle(RGB_img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    cv2.imwrite("./00010200_" + str(len(center)) + ".jpg", RGB_img)

    # 图的可视化
    # 1. 确定每个中心的在 graph 中的位置
    pos_dict = {}
    for i in range(len(center)):
        pos_dict[str(i)] = [center[i][0][1], w - center[i][0][0]]

    fig, ax = plt.subplots()
    nx.draw(g, ax=ax, pos=pos_dict, with_labels=False, width=0.2, edge_color='limegreen', node_color='black', node_size=0.5)  # 设置颜色
    #plt.savefig("./SLIC_graph.svg")
    #plt.show(block=True)
