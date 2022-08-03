# -*- coding: utf-8 -*-
"""
SRCNN: Image Super-Resolution Using Deep Convolutional Networks
Ref: https://arxiv.org/pdf/1501.00092v3.pdf

@author: Pu Ren
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 




class SRCNN(nn.Module):
    """
    Parameters
    ----------
    upscale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.

    """
    def __init__(self, upscale_factor):

        super(SRCNN, self).__init__()

        self.upsacle_factor = upscale_factor
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor

        """
        
        # bicubic upsampling
        x = F.interpolate(x, scale_factor=[self.upsacle_factor,self.upsacle_factor], 
                                      mode='bicubic', align_corners=True)
        
        # CNN extracting features
        x = self.model(x)

        return x