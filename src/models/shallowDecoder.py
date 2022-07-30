#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:54:45 2022

@author: ben
"""

import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init


class shallowDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(shallowDecoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.l1 = nn.Linear(self.input_size[0]*self.input_size[1], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, self.output_size[0]*self.output_size[1])

    def forward(self, x):
        t,c,m,n = x.shape
        x = x.view(t,-1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = x + F.relu(self.l3(x))
        x = self.l4(x) 
        return x.view(t,c,self.output_size[0],self.output_size[1])


class shallowDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(shallowDecoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        #self.l1 = nn.Linear(self.input_size[0]*self.input_size[1], 256)
        #self.l2 = nn.Linear(256, 256)
        #self.l3 = nn.Linear(256, 1024)
        #self.l4 = nn.Linear(10070, self.output_size[0]*self.output_size[1])
        
        c=6
        self.l1 = nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.l2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.l3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, dilation=1)
        self.l4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=1, dilation=1)

        #(input_n - 1)*stride + dilation*(kernelsize -1) + 1
        #(512-1) * 150  + 1*(5-1) +1

    def forward(self, x):
        t,c,m,n = x.shape
        #x = x.view(t,c,-1)
        #print(x.shape)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        x = self.l4(x) 
        #print(x.shape)
        x = x[:,:,0:self.output_size[0], 0:self.output_size[1]]
        return x



class shallowDecoderV2(nn.Module):
    def __init__(self, input_size, output_size):
        super(shallowDecoderV2, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        p = self.input_size[0]*self.input_size[1]
        self.l1 = nn.Linear(p, 2048*2)
        self.l2 = nn.Linear(2048*2, 2048*4)
        self.l3 = nn.Linear(2048*4, 2048*4)
        self.l4 = nn.Linear(2048*4, self.output_size[0]*self.output_size[1])

    def forward(self, x):
        t,c,m,n = x.shape
        x = x.view(t,-1)
        
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = x + F.relu(self.l3(x))
        x = self.l4(x) 
        return x.view(t,c,self.output_size[0],self.output_size[1])
