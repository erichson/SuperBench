# -*- coding: utf-8 -*-
"""
VDSR: Accurate Image Super-Resolution Using Very Deep Convolutional Networks
Ref: https://arxiv.org/abs/1511.04587
    
@author: Pu Ren
"""

import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F 


class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        self.mean = torch.Tensor(mean).view(1, 1, 1, 1)
        self.std = torch.Tensor(std).view(1, 1, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif mode == 'add':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        else:
            raise NotImplementedError


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class VDSR(nn.Module):
    def __init__(self, upscale_factor):
        super(VDSR, self).__init__()
        
        self.upscale_factor = upscale_factor
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
    
        self.subtract_mean = True
        self.shift_mean = ShiftMean(torch.Tensor([-1.3423e-05]), torch.Tensor([0.0024])) # isoflow
        # isoflow: tensor([0.8709]), tensor([0.9469]); the dataset for s8, [1.3754], [1.0246]
        # double gyre: tensor([-1.3423e-05]), tensor([0.0024])
        # rbc: tensor([5.9464]),tensor([21.6621])
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # bicubic upsampling
        x = F.interpolate(x, scale_factor=[self.upscale_factor,self.upscale_factor], 
                                      mode='bicubic', align_corners=True)
        if self.subtract_mean:
            x = self.shift_mean(x, mode='sub')
        
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        
        if self.subtract_mean:
            out = self.shift_mean(out, mode='add')
        
        return out