# -*- coding: utf-8 -*-
# @Time    : 10/14/19 12:40 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : dataset.py
# @Software: PyCharm

from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from config import *
import random

from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from config import *
import random
import scipy.io as io
from densityMapGeneration import gaussian_filter_density

class CSRNetDataset(Dataset):
    '''
    crowdDataset
    '''

    def __init__(self, dataset=DATASET,phase="train", gt_downsample=8):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.phase=phase
        self.name = dataset
        self.img_root = os.path.join(HOME,self.name,"%s_data/images"%(phase))
        self.gt_dmap_root = os.path.join(HOME,self.name,"%s_data/density_maps"%(phase))
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(self.img_root) \
                          if os.path.isfile(os.path.join(self.img_root, filename))]
        self.n_samples = len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name = self.img_names[index]
        img = plt.imread(os.path.join(self.img_root, img_name)) / 255.0
        if len(img.shape) == 2:  # expand grayscale image to three channel.
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), 2)

        gt_dmap = np.load(os.path.join(self.gt_dmap_root, img_name.replace('.jpg', '.npy')))

        # if self.phase=="train":
        #     img, gt_dmap = random_crop(img, gt_dmap, CROP_DOWNSAMPLE)

        if self.gt_downsample > 1:  # to downsample image and density-map to match deep-model.
            ds_rows = int(img.shape[0] // self.gt_downsample)
            ds_cols = int(img.shape[1] // self.gt_downsample)
            img = cv2.resize(img, (ds_cols * self.gt_downsample, ds_rows * self.gt_downsample), interpolation=cv2.INTER_CUBIC)
            gt_dmap = cv2.resize(gt_dmap, (ds_cols, ds_rows), interpolation=cv2.INTER_CUBIC)
            gt_dmap = gt_dmap[np.newaxis, :, :] * self.gt_downsample * self.gt_downsample

        img = img.transpose((2, 0, 1))
        img_tensor = torch.Tensor(img)
        gt_dmap_tensor = torch.Tensor(gt_dmap)
        #print(gt_dmap_tensor.sum())

        return img_tensor, gt_dmap_tensor


def random_crop(image, density, downsample = 4):
    crop_shape = (int(image.shape[0] // downsample), int(image.shape[1] // downsample))

    left_top_x = random.randint(0,image.shape[0]-crop_shape[0])
    left_top_y = random.randint(0, image.shape[1] - crop_shape[1])

    image = image[left_top_x:left_top_x+crop_shape[0],left_top_y:left_top_y+crop_shape[1],:]
    density = density[left_top_x:left_top_x + crop_shape[0], left_top_y:left_top_y+crop_shape[1]]

    return image, density

def random_crop_density(image, points, downsample = 4):
    crop_shape = (int(image.shape[0] // downsample), int(image.shape[1] // downsample))

    left_top_x = random.randint(0,image.shape[0]-crop_shape[0])
    left_top_y = random.randint(0, image.shape[1] - crop_shape[1])

    image = image[left_top_x:left_top_x+crop_shape[0],left_top_y:left_top_y+crop_shape[1],:]

    for i,point in enumerate(points[::-1]):
        if (point[1] < left_top_y) and (point[1] > left_top_y + crop_shape[1]) and \
            (point[0] < left_top_x) and (point[0] > left_top_x + crop_shape[0]):
            points.pop(i)

    #density = density[left_top_x:left_top_x + crop_shape[0], left_top_y:left_top_y+crop_shape[1]]

    return image, points

def generate_density(image_path):
    mat_path = image_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_')

    img = plt.imread(image_path)
    mat = io.loadmat(mat_path)
    #k = np.zeros((img.shape[0], img.shape[1]))
    points = mat["image_info"][0, 0][0, 0][0]
    img, points = random_crop_density(img,points,2)
    density = gaussian_filter_density(img, points)
    return img, density


if __name__ == "__main__":
    img, density = generate_density("/home/zhongyuan/datasets/ShanghaiTech/part_A_final/train_data/images/IMG_2.jpg")
    print(img.shape,density.shape,density.sum())

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cv2.imwrite("sample/image.png",img)

    gt_dmap = density.squeeze()
    shape = gt_dmap.shape
    plt.figure(figsize=(shape[1]/100,shape[0]/100))
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    plt.margins(0,0)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.imshow(gt_dmap)
    plt.savefig("sample/density_map.png")

    # dataset = mcnnDataset(gt_downsample=1)
    # for i, (img, gt_dmap) in enumerate(dataset):
    #     if i <= 100:
    #         continue
    #     img = np.transpose(img.numpy(), (1,2,0))
    #
    #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #
    #     cv2.imwrite("sample/image.png",img)
    #
    #     gt_dmap = gt_dmap.squeeze()
    #     shape = gt_dmap.numpy().shape
    #     plt.figure(figsize=(shape[1]/100,shape[0]/100))
    #     plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
    #     plt.margins(0,0)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis("off")
    #     plt.imshow(gt_dmap)
    #     plt.savefig("sample/density_map.png")
    #     #print(img.shape, gt_dmap.shape)
    #     exit(0)
