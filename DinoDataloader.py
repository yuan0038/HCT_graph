# -*- coding: utf-8 -*-
# @Time : 2023/4/13 10:17 AM
# @Author : Li Zhany
# @Email : 949777411@qq.com
# @File : DinoDataloader.py
# @Project : HCT_graph
from multiprocessing import Manager, Process

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils
from img2graph import img2graph
from torchvision import transforms
import  matplotlib.pyplot as plt
from visual_graph import visual_graph

def softmax(data,t=1):
    exp_data = np.exp(data/t)
    return exp_data/np.sum(exp_data)
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
        self.global_nodes = args.global_nodes # 全局图像采样的节点数目
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
        #img = img.resize((224,224))
        img_list = self.transform(img)  # list: [<PIL.Image.Image image mode=RGB size=224x224>×2,<PIL.Image.Image image mode=RGB size=96x96>×8]
        # 可以将list用于可视化
        crops = []
        labels =[]
        for num,img in enumerate(img_list):
            img = np.array(img)
            nodes, _ = img2graph(img)
            nodes_num = nodes.shape[0]
            purity = nodes[:, -1]
            p = softmax(1-purity,t=2)
            idx = np.arange(0,nodes_num)
            sample_num = self.global_nodes if num<2 else self.local_nodes
            sample = np.random.choice(idx,sample_num,p=p,)
            nodes_sample = nodes[sample]
            #
            nodes_sample = torch.as_tensor(nodes_sample,dtype=torch.float32)
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
            #visual_graph(img,nodes_sample,f'{i}')
        return crops,label


    def __len__(self):
        return len(self.label_list)


class GraphDataset_offline(Dataset):
    def __init__(self, args,mode):
        self.data_path = args.data_path
        self.data_path_list, self.label_list, self.classes_dict = self.get_data_info(self.data_path)


        self.global_nodes = args.global_nodes # 全局图像采样的节点数目
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
                global_crops=np.array(global_crops).tolist()
                local_crops = np.array(local_crops).tolist()
            crops =global_crops+local_crops
            crops= [torch.as_tensor(np.asarray(x),dtype=torch.float32) for x in crops]
            print([np.array(i).shape for i in crops])
            return crops, label
        else:
           # print(file_path)
            with h5py.File(file_path) as f:
                global_crops = f['global_crops'][0]
                global_crops = torch.as_tensor(np.asarray(global_crops),dtype=torch.float32)
                return global_crops,label



    def __len__(self):
        return len(self.label_list)

class GraphDataset_offline_generate_data(Dataset):
    # 用来离线保存数据的数据集
    def __init__(self, args,type='train'):
        self.data_path = args.data_path
        self.data_path_list, self.label_list, self.classes_dict = self.get_data_info(self.data_path)
        self.type = type
        if self.type =='train':
            self.transform = DataAugmentationDINO(
                args.global_crops_scale,
                args.local_crops_scale,
                args.local_crops_number,
            )
            self.global_nodes = args.global_nodes # 全局图像采样的节点数目
            self.local_nodes = args.local_nodes
        else:
            self.transform  = transforms.Compose([
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
        #img = img.resize((224,224))
        img_list = self.transform(img)  # list: [<PIL.Image.Image image mode=RGB size=224x224>×2,<PIL.Image.Image image mode=RGB size=96x96>×8]
        # 可以将list用于可视化
        if self.type =='train':
            crops = []
            labels =[]
            for num,img in enumerate(img_list):
                img = np.array(img)
                nodes, _ = img2graph(img)
                nodes_num = nodes.shape[0]
                purity = nodes[:, -1]
                p = softmax(1-purity,t=2)
                idx = np.arange(0,nodes_num)
                sample_num = self.global_nodes if num<2 else self.local_nodes
                sample = np.random.choice(idx,sample_num,p=p,)
                nodes_sample = nodes[sample]
                #
                #nodes_sample = torch.as_tensor(nodes_sample,dtype=torch.float32)
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
            #visual_graph(img,nodes_sample,f'{i}')
                return crops,label
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
            return nodes_sample,label



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
    lenPerSlice = int(train_data_num/core_num)+1
    file_path_list_slice = []
    out_path_list_slice = []
    file_idx_list= range(train_data_num)
    for i in range(core_num):
        file_path_list_slice.append(file_idx_list[i*lenPerSlice:(i+1)*lenPerSlice])
        out_path_list_slice.append(out_path_list[i*lenPerSlice:(i+1)*lenPerSlice])

    manager = Manager()
    jobs = []

    for i in range(core_num):
        p = Process(target=thread, args=(file_path_list_slice[i], out_path_list_slice[i], train_dataset))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print("all thread_finished")
def thread(file_idx_list,out_path_list,train_dataset:GraphDataset_offline_generate_data):
    data_num = len(file_idx_list)
    for i in range(data_num):
        idx = file_idx_list[i]
        save_path = out_path_list[i]
        print(f"idx {idx}")
        crops,label = train_dataset.__getitem__(idx)
        with h5py.File(save_path, 'w') as f:
            global_crops = np.asarray(crops[:2])
            local_crops = np.asarray(crops[2:])
            f['global_crops'] = global_crops
            f['local_crops'] = local_crops
        f.close()
    print("thread finished")




        # crops,label,file_path = train_dataset.__getitem__(i)
        # file_dir,file_name = file_path.split("/")[-2], file_path.split("/")[-1]
        # #print(file_dir,file_name)
        # file_dir = Path(args.out_path+'/'+file_dir)
        # file_name = file_name.replace('png','h5')
        # Path.mkdir(file_dir,exist_ok=True,parents=True)
        # file_save_path = str(file_dir)+'/'+file_name
        # with h5py.File(file_save_path,'w') as f:
        #     global_crops = np.asarray(crops[:2])
        #     local_crops = np.asarray(crops[2:])
        #     f['global_crops']= global_crops
        #     f['local_crops'] =local_crops
        #     f['label'] = label
        # f.close()

        #print(file_path)

# def calculate_mean_std(dataset):
#         """
#
#         :param dataset: 训练集
#         :param img_size:  用于 将计算得到的mean和std
#         前两维度（聚类中心位置）替换成图像大小的一半
#         :return: mean and std after replace
#         """
#
#         # 训练数据集
#         train_dataset = dataset
#
#         trainloader = DataLoader(train_dataset, batch,                                                                                                                                                                  _size=1, num_workers=0,
#                                      shuffle=False)
#         length = len(train_dataset)
#         std_sum, feats_sum, num_batches = 0, 0, 0
#         for data in tqdm(trainloader):
#             data=data[0]
#             mean_sample = torch.mean(data, dim=0, keepdim=True)
#             feats_sum += mean_sample
#             sqrt_sample = torch.mean(data ** 2, dim=0, keepdim=True)
#
#             std_sample = (sqrt_sample - mean_sample ** 2) ** 0.5
#             if np.isnan(std_sample).any():
#                 print(data[:, 9])
#                 print(std_sample)
#                 print('sqrt', sqrt_sample)
#                 print('mean', mean_sample)
#                 print('mean suqare', mean_sample ** 2)
#                 print(sqrt_sample - mean_sample ** 2)
#                 return
#             std_sum += std_sample
#         # print('MEAN SUM',feats_sum)
#         # print("std sum",std_sum)
#         mean = feats_sum / length  # 均值
#         std = std_sum / length  # 标准差
#         torch.set_printoptions(threshold=torch.inf)
#         print(mean.shape)
#         # mean[0][:2] = img_size / 2
#         # std[0][:2] = img_size / 2
#         print('*' * 40)
#         print("mean:", mean)
#         print("std", std)
#         print("*" * 40)
#         return mean, std
if __name__ == '__main__':
    import yaml
    # config_path = '/home/ubuntu/lxd-workplace/lzy/FewShotLearning/HCT_graph/config.yaml'
    # with open(config_path,'r') as stream:
    #     config = yaml.safe_load(stream)
    # path_params = config["path"]
    from argparse import ArgumentParser
    import warnings

    warnings.filterwarnings("ignore")
    parser = ArgumentParser()
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
           Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
           recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
           local views to generate. Set this parameter to 0 to disable multi-crop training.
           When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
           Used for small local view cropping of multi-crop.""")
    parser.add_argument('--global_nodes',type=int,default=1000)
    parser.add_argument('--local_nodes',type=int,default=250)
    parser.add_argument('--data_path', type=str,
                        default='/home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini_Imagenet_Multicrop/val')
    #/home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini-ImageNet/train_train
    parser.add_argument('--out_path',type=str,default='/home/ubuntu/lxd-workplace/lzy/FewShotLearning/dataset/Mini_Imagenet_Multicrop/val')
    args = parser.parse_args()
    # dataset = GraphDataset_offline(args,'val')
    # l,label = dataset.__getitem__(0)
    # print(l.shape)
    #generate_multicrop_graph_data(args)
    dataset = GraphDataset_offline(args,'train')
    dataset.__getitem__(0)
    # dataset = GraphDataset(args)
    # l,label=dataset.__getitem__(0)

    #calculate_mean_std(dataset,)



    #img  = Image.open("SLIC.jpg")

    # for i in range(len(dataset)):
    #     nodes += dataset.__getitem__(i)
    # print(nodes/len(dataset))
    # 25000