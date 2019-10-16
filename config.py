# -*- coding: utf-8 -*-
# @Time    : 10/14/19 12:39 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : config.py
# @Software: PyCharm


HOME = "/home/zhongyuan/datasets/ShanghaiTech"

DATASET = "part_A_final"
SAVE_PATH = "SGD"
RESUME = True

BATCH_SIZE = 1
MOMENTUM = 0.95
LEARNING_RATE = 1e-7
MAX_EPOCH = 2000
STEPS = (2000,2001)

CROP_DOWNSAMPLE = 4


CSRNET_CONFIG = [[(512,1),(512,1),(512,1),(256,1),(128,1),(64,1)],
                 [(512,2),(512,2),(512,2),(256,2),(128,2),(64,2)],
                 [(512,2),(512,2),(512,2),(256,4),(128,4),(64,4)],
                 [(512,4),(512,4),(512,4),(256,4),(128,4),(64,4)]
                 ]