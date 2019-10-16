# -*- coding: utf-8 -*-
# @Time    : 10/14/19 12:41 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : net.py
# @Software: PyCharm

import torch.nn as nn
from torchvision import models
from config import *
import torch
import os

def conv(in_channel, out_channel, kernel_size, dilation=1, bn=False):
    #padding = 0
    # if kernel_size % 2 == 1:
    #     padding = int((kernel_size - 1) / 2)
    padding = dilation # maintain the previous size
    if bn:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,),
            nn.BatchNorm2d(out_channel, momentum=0.005),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, dilation=dilation,),
            # nn.BatchNorm2d(out_channel, momentum=0.005),
            nn.ReLU(inplace=True)
        )


class CSRNet(nn.Module):
    def __init__(self, phase = 1):
        super(CSRNet,self).__init__()

        self.net_config = CSRNET_CONFIG[phase]

        self.front_end = nn.Sequential(conv(3, 64, 3),
                                       conv(64, 64, 3),
                                       nn.MaxPool2d(2, 2),
                                       conv(64, 128, 3),
                                       conv(128, 128, 3),
                                       nn.MaxPool2d(2, 2),
                                       conv(128, 256, 3),
                                       conv(256, 256, 3),
                                       conv(256, 256, 3),
                                       nn.MaxPool2d(2, 2),
                                       conv(256, 512, 3),
                                       conv(512, 512, 3),
                                       conv(512, 512, 3)
                                       )
        self.back_end = []
        for i,config in enumerate(self.net_config):
            if i == 0:
                self.back_end.append(conv(512, config[0],3,dilation=config[1]))
            else:
                self.back_end.append(conv(self.net_config[i-1][0], config[0], 3, dilation=config[1]))

        self.back_end = nn.Sequential(*self.back_end)
        self.conv1 = nn.Conv2d(64, 1, 1)
        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("loading pretrained vgg16!")
        if os.path.exists("weights/vgg16.pth"):
            print("find pretrained weights!")
            vgg16 = models.vgg16(pretrained=False)
            vgg16_weights = torch.load("weights/vgg16.pth")
            vgg16.load_state_dict(vgg16_weights)
        else:
            vgg16 = models.vgg16(pretrained=True)

        for i in range(len(self.front_end.state_dict().items())):
            #print(list(self.front_end.state_dict().items())[i])
            list(self.front_end.state_dict().items())[i][1].data[:] = list(vgg16.state_dict().items())[i][1].data[:]
            #print("\n\n\n\n")
            #print(list(self.front_end.state_dict().items())[i])
            #exit(0)

    def forward(self, x):
        x = self.front_end(x)
        x = self.back_end(x)
        x = self.conv1(x)
        return x

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    net = CSRNet()
    #print(net.front_end.state_dict())
    x = torch.ones((1, 3, 256, 256))
    print(x.size())
    y = net(x)
    print(y.size())