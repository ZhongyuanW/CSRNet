# -*- coding: utf-8 -*-
# @Time    : 10/15/19 10:59 AM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : train.py
# @Software: PyCharm

import net as Net
import dataset as Dataset
import torch.utils.data.dataloader as Dataloader
import torch.nn as nn
import torch.optim as optim
import time
import torch
import visdom
from torch.autograd import Variable
from config import *
import sys
import numpy as np
#import eval as eval
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

viz = visdom.Visdom(env="CSRNet")

def train():

    dataset = Dataset.CSRNetDataset(gt_downsample=8)
    dataloader = Dataloader.DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True,worker_init_fn=worker_init_fn)
    print("dataset size is: %d"%dataset.__len__())

    test_dataset = Dataset.CSRNetDataset(phase="test",gt_downsample=8)
    test_dataloader = Dataloader.DataLoader(test_dataset,batch_size=1,
                        shuffle=True, drop_last=True,worker_init_fn=worker_init_fn)

    net = Net.CSRNet()
    net = nn.DataParallel(net)
    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    #optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='sum').cuda()

    t0 = time.time()
    start_epoch = 0
    step_index = 0

    min_mae = sys.maxsize
    min_epoch = -1
    epoch_list = []
    train_loss_list = []
    epoch_loss_list = []
    test_mae_list = []

    if RESUME:
        path_list = os.listdir("weights/%s"%SAVE_PATH)
        ep_list = [int(i.split("_")[2]) for i in path_list]
        curr_index = ep_list.index(max(ep_list))
        start_epoch = step_index = ep_list[curr_index]
        weight_path = os.path.join("weights/%s"%SAVE_PATH,path_list[curr_index])
        weight = torch.load(weight_path)
        net.load_state_dict(weight)
        print("resume weight %s, at %d\n" % (weight_path,start_epoch))
        min_mae = float(path_list[curr_index].split("_")[-1].split(".")[0])/100
        min_epoch = start_epoch

    for i in range(start_epoch+1, MAX_EPOCH):

        if i in STEPS:
            step_index += 1
            adjust_learning_rate(optimizer, 0.1, step_index)

        ## train ##
        epoch_loss = 0
        net.train()
        for _,(images,targets) in enumerate(dataloader):

            images,targets = Variable(images),Variable(targets)
            images,targets = images.cuda(),targets.cuda()

            optimizer.zero_grad()

            densitymaps = net(images)

            loss = criterion(densitymaps,targets)
            epoch_loss += min(loss.item(),200)

            loss.backward()
            optimizer.step()

        epoch_loss_list.append(epoch_loss)
        train_loss_list.append(epoch_loss/len(dataloader))
        epoch_list.append(i)

        print("train [%d/%d] timer %.4f, loss %.4f"%(i,MAX_EPOCH,time.time()-t0,epoch_loss/len(dataloader)))
        t0 = time.time()

        ## eval ##
        net.eval()
        mae = 0

        for _,(images,targets) in enumerate(test_dataloader):
            images, targets = Variable(images), Variable(targets)
            images, targets = images.cuda(), targets.cuda()

            densitymaps = net(images)

            mae += abs(densitymaps.data.sum()-targets.data.sum()).item()

        mae = mae / len(test_dataloader)

        if(mae<min_mae):
            min_mae = mae
            min_epoch = i
            print("save state, epoch: %d" % i)
            torch.save(net.state_dict(), "weights/%s/CSRNet_epoch_%d_%d.pth" % (SAVE_PATH,i,mae*100))
        test_mae_list.append(mae)
        print("eval [%d/%d] mae %.4f, min_mae %.4f, min_epoch %d\n"%(i,MAX_EPOCH,mae,min_mae, min_epoch))

        ## vis ##
        viz.line(win="1", X=epoch_list, Y=train_loss_list, opts=dict(title="train_loss"))
        viz.line(win="2", X=epoch_list, Y=test_mae_list, opts=dict(title="test_mae"))

        index = random.randint(0,len(test_dataloader)-1)
        image,gt_map = test_dataset[index]
        viz.image(win="3",img=image,opts=dict(title="test_image"))
        viz.image(win="4",img=gt_map/(gt_map.max())*255,opts=dict(title="gt_map_%.4f"%(gt_map.sum())))

        image = Variable(image.unsqueeze(0)).cuda()
        densitymap = net(image)
        densitymap = densitymap.squeeze(0).detach().cpu().numpy()
        viz.image(win="5",img=densitymap/(densitymap.max())*255,opts=dict(title="predictImages_%.4f"%(densitymap.sum())))

def adjust_learning_rate(optimizer, gamma, step):
    lr = LEARNING_RATE * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def setup_seed(seed=2019):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id): # After creating the workers, each worker has an independent seed that is initialized to the curent random seed + the id of the worker
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    setup_seed()
    train()