# -*- coding: utf-8 -*-
"""
Classic bicubic interpolation

@author: Pu Ren
"""

import torch.nn as nn
import torch.nn.functional as F 


class Bicubic(nn.Module):
    """
    Parameters
    ----------
    upscale_factor : int
    """
    
    def __init__(self, upscale_factor):

        super(Bicubic, self).__init__()

        self.upsacle_factor = upscale_factor
        
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
        
        return x