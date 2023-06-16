#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:54:45 2022

@author: ben
"""

import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init


class subpixelCNN(nn.Module):
    def __init__(self, in_feats, upscale_factor=4, width=1):
        super(subpixelCNN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_feats, 128*width, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(128*width, 128*width, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(128*width, 64*width, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64*width, in_feats * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)