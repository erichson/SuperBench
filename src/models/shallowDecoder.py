#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:54:45 2022

@author: ben
"""

import torch.nn as nn
import torch.nn.functional as F

class shallowDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(shallowDecoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.l1 = nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.l2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.l3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, dilation=1)
        self.l4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, dilation=1)

    def forward(self, x):
        t,c,m,n = x.shape
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x) 
        #print(x.shape)
        x = x[:,:,0:self.output_size[0], 0:self.output_size[1]]
        return x
