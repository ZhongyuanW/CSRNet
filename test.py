# -*- coding: utf-8 -*-
# @Time    : 10/15/19 11:30 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : test.py
# @Software: PyCharm

import torch
from config import *
import os
import torch.nn as nn
import net as Net
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

def test(image_path = os.path.join(HOME,"part_A_final/test_data/images","IMG_6.jpg"),
         weight_path = "weights/SGD/CSRNet_epoch_543_7503.pth"):
    net = Net.CSRNet()
    net = nn.DataParallel(net)
    net = net.cuda()
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    print("load weight completed!")
    net.eval()

    if not os.path.exists(image_path):
        print("not find image path!")
        exit(-1)
    image = cv2.imread(image_path)
    if len(image.shape) == 2:  # expand grayscale image to three channel.
        image = image[:, :, np.newaxis]
        image = np.concatenate((image, image, image), 2)
    image = image.transpose((2, 0, 1))
    img_tensor = torch.Tensor(image)
    image = Variable(img_tensor.unsqueeze(0)).cuda()

    gt_dmap_root = os.path.join(HOME, "part_A_final", "test_data/density_maps")
    gt_dmap = np.load(os.path.join(gt_dmap_root, image_path.split("/")[-1].replace('.jpg', '.npy')))

    density_map = net(image).squeeze(0).cpu().data

    crowd_counting = density_map.sum()
    return crowd_counting,gt_dmap.sum(),density_map


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    count, gt, densitymap = test()
    print("the image has %.4f persons, gt: %.4f"%(count,gt))
    densitymap = densitymap.squeeze(0)
    plt.imsave("res/res.png",densitymap/densitymap.max()*255)
    print("the prec density map save at res/res.png")