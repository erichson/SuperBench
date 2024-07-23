import torch.nn as nn
from .FNO_basic import SpectralConv2d
from .FNO_util import _get_act, add_padding2, remove_padding2
import torch
import torch.nn.functional as F
class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        len_c = mean.shape[0]
        self.mean = torch.Tensor(mean).view(1, len_c, 1, 1)
        self.std = torch.Tensor(std).view(1, len_c, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif mode == 'add':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        else:
            raise NotImplementedError

class FNO2D_conv(nn.Module):
    def __init__(self, modes1, modes2,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=3,
                 act='relu', 
                 pad_ratio=[0., 0.], mean =[0],std=[1],scale_factor = 8):
        super(FNO2D_conv, self).__init__()
        """
        Args:s
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions'
        self.modes1 = modes1
        self.modes2 = modes2
    
        self.pad_ratio = pad_ratio
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv2d(
            in_size, out_size, mode1_num, mode2_num)
            for in_size, out_size, mode1_num, mode2_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        # self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.conv1 = nn.Conv2d(layers[-1],fc_dim,9,1,4)
        # self.fc2 = nn.Linear(fc_dim, layers[-1])
        self.conv2 = nn.Conv2d(fc_dim,layers[-1],5,1,2)
        # self.fc3 = nn.Linear(layers[-1], out_dim)
        self.conv3 = nn.Conv2d(layers[-1],out_dim,5,1,2)
        self.act = _get_act(act)
        self.shiftmean = ShiftMean(torch.Tensor(mean), torch.Tensor(std))
        self.scale_factor = scale_factor
    def forward(self, x):
        '''
        Args:
            - x : (batch size, c, x_grid, y_grid)
        Returns:
            - x: (batch size, c, x_grid, y_grid)
        '''
        size_1, size_2 = x.shape[-2], x.shape[-1]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = F.interpolate(x,scale_factor = self.scale_factor, mode='bicubic') # LR to HR
        x = self.shiftmean(x,"sub")
        x = x.permute(0, 2, 3, 1) # B,C,X,Y to B,X,Y,C
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)   # B,X,Y,C to B, C, X, Y
        # x = add_padding2(x, num_pad1, num_pad2)
        size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        # x = remove_padding2(x, num_pad1, num_pad2)
        # x = x.permute(0, 2, 3, 1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        # x = x.permute(0, 3, 1, 2) # B,X,Y,C to B,C,X,Y
        x = self.shiftmean(x, 'add')
        return x