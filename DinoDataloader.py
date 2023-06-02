# -*- coding: utf-8 -*-
# @Time : 2023/4/13 10:17 AM
# @Author : Li Zhany
# @Email : 949777411@qq.com
# @File : DinoDataloader.py
# @Project : HCT_graph
import random
from multiprocessing import Manager, Process

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

#import utils
from img2graph import img2graph
from torchvision import transforms
import matplotlib.pyplot as plt
from visual_graph import visual_graph


def softmax(data, t=1):
    exp_data = torch.exp(data / t)
    return exp_data / torch.sum(exp_data,axis=0)
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        '''
               mini/tiered:   
                   mean = [0.485, 0.456, 0.406]
                   std = [0.229, 0.224, 0.225]

               cifar/FC100:  
                   mean = [n02099601/255.0 for n02099601 in [129.37731888,  124.10583864, 112.47758569]]
                   std = [n02099601/255.0 for n02099601 in [68.20947949,  65.43124043,  70.45866994]]      

               '''
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),

        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),

        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),

        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


## 用那个啥粒矩的中心，构造固定大小的框，然后按概率选框
class GraphDataset(Dataset):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_path_list, self.label_list, self.classes_dict = self.get_data_info(self.data_path)

        self.transform = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
        self.global_nodes = args.global_nodes  # 全局图像采样的节点数目
        self.local_nodes = args.local_nodes

    def get_data_info(self, data_path):
        classes_dict = {}
        data_path_list = []
        label_list = []
        root_path = Path(data_path)
        for i, class_path in enumerate(root_path.iterdir()):
            class_name = str(class_path).split('/')[-1]
            classes_dict[class_name] = i
            for data in class_path.iterdir():
                data_path_list.append(str(data))
                label_list.append(i)

        return data_path_list, label_list, classes_dict

    def __getitem__(self, idx):
        file_path = self.data_path_list[idx]
        label = self.label_list[idx]
        img = Image.open(file_path)
        # img = img.resize((224,224))
        img_list = self.transform(
            img)  # list: [<PIL.Image.Image image mode=RGB size=224x224>×2,<PIL.Image.Image image mode=RGB size=96x96>×8]
        # 可以将list用于可视化
        crops = []
        labels = []
        for num, img in enumerate(img_list):
            img = np.array(img)
            nodes, _ = img2graph(img)
            nodes_num = nodes.shape[0]
            purity = nodes[:, -1]
            p = softmax(1 - purity, t=2)
            idx = np.arange(0, nodes_num)
            sample_num = self.global_nodes if num < 2 else self.local_nodes
            sample = np.random.choice(idx, sample_num, p=p, )
            nodes_sample = nodes[sample]
            #
            nodes_sample = torch.as_tensor(nodes_sample, dtype=torch.float32)
            crops.append(nodes_sample)
            labels.append(label)
            # cv2.imwrite(f'./{num}.png', img)
            # for i in range(nodes_sample.shape[0]):
            #     y, x, Rx, Ry = int(nodes_sample[i][0]), int(nodes_sample[i][1]), int(nodes_sample[i][2]), int(nodes_sample[i][3])
            #     cv2.rectangle(img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
            # cv2.imwrite(f'./sample_crops_{num}.png', img)
            # for i in range(nodes.shape[0]):
            #     y, x, Rx, Ry = int(nodes[i][0]), int(nodes[i][1]), int(nodes[i][2]), int(nodes[i][3])
            #     cv2.rectangle(img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
            # cv2.imwrite(f'./crops_{num}.png',img)
            # visual_graph(img,nodes_sample,f'{i}')
        return crops, label

    def __len__(self):
        return len(self.label_list)


# 从原图上构图，
# 最后要返回每张图像对应的全局结点和局部结点（所以得确定）
#
class GraphDataset_offline(Dataset):
    def __init__(self, args, mode):
        self.data_path = args.data_path
        self.data_path_list, self.label_list, self.classes_dict = self.get_data_info(self.data_path)

        self.global_nodes = args.global_nodes  # 全局图像采样的节点数目
        self.local_nodes = args.local_nodes
        self.mode = mode

    def get_data_info(self, data_path):
        classes_dict = {}
        data_path_list = []
        label_list = []
        root_path = Path(data_path)
        for i, class_path in enumerate(root_path.iterdir()):
            class_name = str(class_path).split('/')[-1]
            classes_dict[class_name] = i
            for data in class_path.iterdir():
                data_path_list.append(str(data))
                label_list.append(i)

        return data_path_list, label_list, classes_dict

    def __getitem__(self, idx):
        file_path = self.data_path_list[idx]
        label = self.label_list[idx]

        if self.mode == 'train':
            with h5py.File(file_path) as f:
                global_crops = f['global_crops']
                local_crops = f['local_crops']
                global_crops = np.array(global_crops).tolist()
                local_crops = np.array(local_crops).tolist()
            crops = global_crops + local_crops
            crops = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in crops]
            #print([np.array(i).shape for i in crops])
            return crops, label
        else:
            # print(file_path)
            with h5py.File(file_path) as f:
                global_crops = f['global_crops'][0]
                global_crops = torch.as_tensor(np.asarray(global_crops), dtype=torch.float32)
                return global_crops, label

    def __len__(self):
        return len(self.label_list)


class GraphDataset_offline_generate_data(Dataset):
    # 用来离线保存数据的数据集
    def __init__(self, args, type='train'):
        self.data_path = args.data_path
        self.data_path_list, self.label_list, self.classes_dict = self.get_data_info(self.data_path)
        self.type = type
        if self.type == 'train':
            self.transform = DataAugmentationDINO(
                args.global_crops_scale,
                args.local_crops_scale,
                args.local_crops_number,
            )
            self.global_nodes = args.global_nodes  # 全局图像采样的节点数目
            self.local_nodes = args.local_nodes
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
            ])

    def get_data_info(self, data_path):
        classes_dict = {}
        data_path_list = []
        label_list = []
        root_path = Path(data_path)
        for i, class_path in enumerate(root_path.iterdir()):
            class_name = str(class_path).split('/')[-1]
            classes_dict[class_name] = i
            for data in class_path.iterdir():
                data_path_list.append(str(data))
                label_list.append(i)

        return data_path_list, label_list, classes_dict

    def __getitem__(self, idx):
        file_path = self.data_path_list[idx]
        label = self.label_list[idx]
        img = Image.open(file_path)
        # img = img.resize((224,224))
        img_list = self.transform(
            img)  # list: [<PIL.Image.Image image mode=RGB size=224x224>×2,<PIL.Image.Image image mode=RGB size=96x96>×8]
        # 可以将list用于可视化
        if self.type == 'train':
            crops = []
            labels = []
            for num, img in enumerate(img_list):
                img = np.array(img)
                nodes, _ = img2graph(img)
                nodes_num = nodes.shape[0]
                purity = nodes[:, -1]
                p = softmax(1 - purity, t=2)
                idx = np.arange(0, nodes_num)
                sample_num = self.global_nodes if num < 2 else self.local_nodes
                sample = np.random.choice(idx, sample_num, p=p, )
                nodes_sample = nodes[sample]
                #
                # nodes_sample = torch.as_tensor(nodes_sample,dtype=torch.float32)
                crops.append(nodes_sample)
                labels.append(label)
                # cv2.imwrite(f'./{num}.png', img)
                # for i in range(nodes_sample.shape[0]):
                #     y, x, Rx, Ry = int(nodes_sample[i][0]), int(nodes_sample[i][1]), int(nodes_sample[i][2]), int(nodes_sample[i][3])
                #     cv2.rectangle(img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
                # cv2.imwrite(f'./sample_crops_{num}.png', img)
                # for i in range(nodes.shape[0]):
                #     y, x, Rx, Ry = int(nodes[i][0]), int(nodes[i][1]), int(nodes[i][2]), int(nodes[i][3])
                #     cv2.rectangle(img, (x - Rx, y - Ry), (x + Rx, y + Ry), (0, 255, 0), 1)
                # cv2.imwrite(f'./crops_{num}.png',img)
                # visual_graph(img,nodes_sample,f'{i}')
                return crops, label
        else:
            img = np.array(img)
            nodes, _ = img2graph(img)
            nodes_num = nodes.shape[0]
            purity = nodes[:, -1]
            p = softmax(1 - purity, t=2)
            idx = np.arange(0, nodes_num)
            sample_num = self.global_nodes
            sample = np.random.choice(idx, sample_num, p=p, )
            nodes_sample = nodes[sample]
            return nodes_sample, label

    def __len__(self):
        return len(self.label_list)


def generate_multicrop_graph_data(args):
    train_dataset = GraphDataset_offline_generate_data(args)

    train_data_num = len(train_dataset)
    file_path_list = train_dataset.data_path_list

    out_path_list = []
    for i in range(train_data_num):
        file_dir, file_name = file_path_list[i].split("/")[-2], file_path_list[i].split("/")[-1]
        # print(file_dir,file_name)
        file_dir = Path(args.out_path + '/' + file_dir)
        file_name = file_name.replace('png', 'h5')
        Path.mkdir(file_dir, exist_ok=True, parents=True)
        file_save_path = str(file_dir) + '/' + file_name
        out_path_list.append(file_save_path)
    core_num = 20
    lenPerSlice = int(train_data_num / core_num) + 1
    file_path_list_slice = []
    out_path_list_slice = []
    file_idx_list = range(train_data_num)
    for i in range(core_num):
        file_path_list_slice.append(file_idx_list[i * lenPerSlice:(i + 1) * lenPerSlice])
        out_path_list_slice.append(out_path_list[i * lenPerSlice:(i + 1) * lenPerSlice])

    manager = Manager()
    jobs = []

    for i in range(core_num):
        p = Process(target=thread, args=(file_path_list_slice[i], out_path_list_slice[i], train_dataset))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print("all thread_finished")


# Resize Mini-ImageNet from 84x84 to 256x256 and turn into graph,only save nodes at last
class MiniImageNet(Dataset):
    set_types = ['train', 'val', 'test']

    def __init__(self, data_dir, set_type):

        super(MiniImageNet, self).__init__()
        # assert set_type in MiniImageNet.set_types
        # if set_type == 'train':
        #     name = 'train_train'
        # elif set_type == 'val':
        #     name = 'train_val'
        # else:
        #     name = 'test'
        # data_dir = data_dir + '/' + name

        self.data_list, self.label_list = self.get_data_list(data_dir)

    def get_data_list(self, data_dir):
        data_dir = Path(data_dir)
        data_list = []
        label_list = []
        for i, class_dir in enumerate(data_dir.iterdir()):  #
            for data_path in class_dir.iterdir():
                data_list.append(str(data_path))
                label_list.append(i)

        return data_list,label_list
    def __getitem__(self, idx):
        data_path = self.data_list[idx]
        label = self.label_list[idx]
        image = Image.open(data_path)
        image = image.resize((256,256))
        image = np.array(image)
        return image, label
    def __len__(self):
        return len(self.data_list)
def thread(data_slice,out_path_slice):
    data_num = len(data_slice)
    for i in range(data_num):
        img = Image.open(data_slice[i])
        #img = img.resize((256,256))
        img = np.array(img)
        nodes, _ = img2graph(img)
        print(out_path_slice[i])
        with h5py.File(out_path_slice[i], 'w') as f:
            f['nodes']=np.asarray(nodes)
        f.close()
    print('one thread finish')

class MultiprocessGenerateData(object):
    def __init__(self,data_list,out_list,core_num=20):
        data_slices = []
        out_path_slices = []
        data_num = len(data_list)
        num_per_slice = int(data_num/core_num)+1

        for i in range(core_num):
            data_slices.append(data_list[i*num_per_slice:(i+1)*num_per_slice])
            out_path_slices.append(out_list[i*num_per_slice:(i+1)*num_per_slice])

        jobs = []
        for i in range(core_num):
            p = Process(target=thread, args=(data_slices[i],out_path_slices[i]))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        print("all thread_finished")

# =======just run it to generate data
def generate_data_main():
    from argparse import ArgumentParser
    import warnings

    warnings.filterwarnings("ignore")
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini-ImageNet/test')
    # /home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini-ImageNet/train_train
    parser.add_argument('--out_path', type=str,
                        default='/home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini-Imagenet_84_graph/test')
    args = parser.parse_args()

    train_dataset = MiniImageNet(data_dir=args.data_dir, set_type='test')
    data_list = train_dataset.data_list
    out_path = Path(args.out_path)
    Path.mkdir(out_path, parents=True, exist_ok=True)
    out_path = [path.replace("png", 'h5') for path in data_list]
    out_path_list = [path.replace(args.data_dir, args.out_path) for path in out_path]
    out_dir_list = list(set([path.rsplit('/', 1)[0] for path in out_path_list]))
    for out_dir in out_dir_list:
        Path.mkdir(Path(out_dir), exist_ok=True, parents=True)
    MultiprocessGenerateData(data_list, out_list=out_path_list)


class MiniImageNet_Graph(Dataset):
    def __init__(self, dataset_dir, set_type):
        super(MiniImageNet_Graph, self).__init__()
        assert set_type in MiniImageNet.set_types
        self.set_type = set_type
        if set_type == 'train':
            name = 'train_train'
        elif set_type == 'val':
            name = 'val'
        else:
            name = 'test'
        data_dir = dataset_dir + '/' + name

        self.data_list, self.label_list = self.get_data_list(data_dir)
        mean = torch.tensor([4.2000e+01, 4.2000e+01, 1.6550e+00, 2.1434e+00, 1.2089e+02, 7.6961e+02,
        1.1392e+02, 7.3178e+02, 1.0042e+02, 7.2574e+02, 5.8929e-01])
        std = torch.tensor([4.2000e+01, 4.2000e+01, 2.1014e+00, 1.4293e+00, 5.0151e+01, 1.0297e+03,
        4.9215e+01, 9.8820e+02, 4.9191e+01, 1.0142e+03, 2.4146e-01])
        self.normalize =Normalize(mean,std)
    def get_data_list(self, data_dir):
        data_dir = Path(data_dir)
        data_list = []
        label_list = []
        self.class_dict = {}
        for i, class_dir in enumerate(data_dir.iterdir()):
            class_name = str(class_dir).split('/')[-1]
            self.class_dict[class_name]=i
            for data_path in class_dir.iterdir():
                data_list.append(str(data_path))
                label_list.append(i)

        return data_list,label_list

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        label = torch.as_tensor(self.label_list[idx],dtype=torch.long)

        with h5py.File(file_path) as f:
            nodes=f['nodes']
            nodes = torch.tensor(nodes,dtype=torch.float32)

            if self.set_type == 'train':
                purity = nodes[:, -1]
                p = softmax(1 - purity, t=2)
                node_num = nodes.shape[0]
                nodes = self.normalize(nodes)
                return nodes,node_num,np.array(p),label
            else:
                nodes = self.normalize(nodes)
                return nodes,label

    def __len__(self):
        return len(self.data_list)
def My_collate_fn(batched_data):
    # batch_data [(nodes,node_num,probability,label,),]
    data_num = len(batched_data)
    data = list(map(list,zip(*batched_data))) #  [[nodes,],[node_num],...]
    nodes = data[0]
    nodes_num = data[1]
    probability = data[2]
    labels = data[3]
    #print(labels)
    # We fist calculate the average nodes number  of batched img-level-nodes 'avg' to fix the sample nodes in this batch
    # Then we sample  'avg' nodes for each img-level-nodes to replace img-level-nodes
    #print(batched_data)

    nodes_num =np.array(data[1])
    #print(nodes_num)
    node_num_avg = int(np.average(nodes_num))
    sample_nodes_list=[]

    global_level = int(3/4 *node_num_avg)  #  全局结点取原图平均结点的四分之三好吧
    local_level = int(1/4 *node_num_avg)

    for i in range(data_num):
        all_nodes=[]  # to save global nodes and local nodes 2+8
        idx = np.arange(0, nodes_num[i])
        #print(probability[i])
        for j in range(2): # global
            sample_idx=np.random.choice(idx, global_level, p=probability[i], )
            nodes_sampled = nodes[i][sample_idx]
            all_nodes.append(torch.as_tensor(nodes_sampled,dtype=torch.float32))
        for k in range(8): # local
            sample_idx = np.random.choice(idx, local_level, p=probability[i], )
            nodes_sampled = nodes[i][sample_idx]
            all_nodes.append(torch.as_tensor(nodes_sampled, dtype=torch.float32))
        sample_nodes_list.append(all_nodes)

    batched_data=list(zip(*sample_nodes_list))
    batched_data=[torch.stack(i) for i in batched_data]

    return batched_data,torch.as_tensor(labels)
    # puritys = []

def calculate_mean_std(dataset, img_size):
    """

    :param dataset: 训练集
    :param img_size:  用于 将计算得到的mean和std
    前两维度（聚类中心位置）替换成图像大小的一半
    :return: mean and std after replace
    """

    # 训练数据集
    train_dataset = dataset

    trainloader = DataLoader(train_dataset, batch_size=1, num_workers=0,
                                 shuffle=False)
    length = len(train_dataset)
    std_sum, feats_sum, num_batches = 0, 0, 0
    for data in tqdm(trainloader):
        # print(data[0].shape)
        # return
        x = data[0]
        mean_sample = torch.mean(x, dim=1, keepdim=True)
        feats_sum += mean_sample
        sqrt_sample = torch.mean(x ** 2, dim=1, keepdim=True)

        std_sample = (sqrt_sample - mean_sample ** 2) ** 0.5
        if np.isnan(std_sample).any():
            print(x[:, 9])
            print(std_sample)
            print('sqrt', sqrt_sample)
            print('mean', mean_sample)
            print('mean suqare', mean_sample ** 2)
            print(sqrt_sample - mean_sample ** 2)
            return
        std_sum += std_sample
    # print('MEAN SUM',feats_sum)
    # print("std sum",std_sum)
    mean = feats_sum / length  # 均值
    std = std_sum / length  # 标准差
    mean = np.squeeze(mean)
    std = np.squeeze(std)
    torch.set_printoptions(threshold=torch.inf)

    mean[:2] = img_size/2
    std[:2] = img_size/2
    print('*' * 40)
    print("mean:", mean)
    print("std", std)
    print("*" * 40)
    return mean, std


class Normalize:
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    def __call__(self, nodes:torch.Tensor):
        assert isinstance(nodes,torch.Tensor)
        return nodes.sub_(self.mean).div_(self.std)
def check():
    dataset_dir ='/home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini-Imagenet_84_graph/'
    train_dataset = MiniImageNet_Graph(dataset_dir,set_type='train')
    print(len(train_dataset))

    # calculate_mean_std(train_dataset,84)
    # return
    dataloader = DataLoader(train_dataset,batch_size=2,collate_fn=My_collate_fn)
    for batch_data in dataloader:
        print(batch_data[0])
        return
if __name__ == '__main__':
    import yaml
    # config_path = '/home/ubuntu/lxd-workplace/lzy/FewShotLearning/HCT_graph/config.yaml'
    # with open(config_path,'r') as stream:
    #     config = yaml.safe_load(stream)
    # path_params = config["path"]
    #generate_data_main()
    check()
    #generate_data_main()
    #print(out_path[0])

   # train_dataset.__getitem__(0)
    # # calculate_mean_std(dataset,)
    # print(len(train_dataset))
    # MultiprocessGenerateData(train_dataset.data_list)
    # img  = Image.open("SLIC.jpg")

    # for i in range(len(dataset)):
    #     nodes += dataset.__getitem__(i)
    # print(nodes/len(dataset))
    # 25000
