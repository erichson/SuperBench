import torch
from torchinfo import summary
from src.models import *

input_size = [360, 360]
output_size = [360, 360]
upscale_factor = 1
width = 64
in_channels = 3
out_channels = 3

model_list = {
        'shallowDecoder': shallowDecoder(output_size, upscale_factor=upscale_factor),
        'shallowDecoderMultiChan': shallowDecoder(output_size, upscale_factor=upscale_factor, in_channels=in_channels, out_channels=out_channels),
        'subpixelCNN': subpixelCNN(upscale_factor=upscale_factor, width=width)
}

model = model_list['shallowDecoderMultiChan']
summary(model, input_size=(2, in_channels, input_size[0], input_size[1]))
