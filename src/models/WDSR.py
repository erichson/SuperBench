# -*- coding: utf-8 -*-
"""
WDSR: Wide Activation for Efficient and Accurate Image Super-Resolution
Ref: https://arxiv.org/abs/1808.08718

@author: Pu Ren
"""

from torch import nn
from torch.nn.utils import weight_norm
import torch


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


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=3, padding=1, padding_mode='circular')),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1, padding_mode='circular'))
        )

    def forward(self, x):
        return x + self.module(x) * self.res_scale


class WDSR(nn.Module):
    def __init__(self, upscale_factor):
        super(WDSR, self).__init__()
        
        self.n_feats = 32
        self.expansion_ratio = 4
        self.res_scale = 0.1
        self.n_res_blocks = 18
        self.scale = upscale_factor
        
        head = [weight_norm(nn.Conv2d(1, self.n_feats, kernel_size=3, padding=1, padding_mode='circular'))]
        body = [ResBlock(self.n_feats, self.expansion_ratio, self.res_scale) for _ in range(self.n_res_blocks)]
        tail = [weight_norm(nn.Conv2d(self.n_feats, 1 * (self.scale ** 2), kernel_size=3, padding=1, padding_mode='circular')),
                nn.PixelShuffle(self.scale)]
        skip = [weight_norm(nn.Conv2d(1, 1 * (self.scale ** 2), kernel_size=5, padding=2, padding_mode='circular')), 
                nn.PixelShuffle(self.scale)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        self.subtract_mean = True
        self.shift_mean = ShiftMean(torch.Tensor([5.9464]), torch.Tensor([21.6621])) # isoflow
        # isoflow: tensor([0.8709]), tensor([0.9469]); the dataset for s8, [1.3754], [1.0246]
        # double gyre: tensor([-1.3423e-05]), tensor([0.0024])
        # rbc: tensor([5.9464]),tensor([21.6621])

    def forward(self, x):
        
        # input size: [N,C,H,W]
        
        if self.subtract_mean:
            x = self.shift_mean(x, mode='sub')

        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s

        if self.subtract_mean:
            x = self.shift_mean(x, mode='add')

        return x



