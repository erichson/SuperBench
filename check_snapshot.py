'''training function'''

import numpy as np
import torch
from torch import nn
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from src.models import *
from src.data_loader_crop import getData
from utils import *
import neptune
import random
def main():
    parser = argparse.ArgumentParser(description='training parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='nskt_16k', help='dataset')
    parser.add_argument('--data_path', type=str, default='../superbench/datasets/nskt16000_1024', help='the folder path of dataset')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size for high-resolution snapshots')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')    
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--model_path', type=str, default='results/model_EDSR_sst4_0.0001_5544.pt', help='saved model')
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'), help='load the pretrained model')
    
    # arguments for training
    parser.add_argument('--model', type=str, default='subpixelCNN', help='model')
    parser.add_argument('--epochs', type=int, default=300, help='max epochs')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--step_size', type=int, default=1000, help='step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.97, help='coefficient for scheduler')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

    # arguments for model
    parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
    parser.add_argument('--in_channels', type=int, default=2, help='num of input channels')
    parser.add_argument('--hidden_channels', type=int, default=32, help='num of hidden channels')
    parser.add_argument('--out_channels', type=int, default=2, help='num of output channels')
    parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')
    parser.add_argument('--loss_type', type=str, default='l1', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='type of scheduler')

    args = parser.parse_args()
    print(args)
    # % --- %
    # Set random seed to reproduce the work
    # % --- %
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # % --- %
    # Load data
    # % --- %
    resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name) # 
    train_loader, val1_loader, val2_loader, test1_loader, test2_loader = getData(args, args.n_patches, std=std)
    print('The data resolution is: ', resol)
    print("mean is: ",mean)
    print("std is: ",std)

    # % --- %
    # Get model
    # % --- %
    # some hyper-parameters for SwinIR
    upscale = args.upscale_factor
    window_size = 8
    height = (args.crop_size // upscale // window_size + 1) * window_size
    width = (args.crop_size // upscale // window_size + 1) * window_size

    for idx,(lr,hr) in enumerate(train_loader):
        
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(hr[3,0,:,:])
        ax[0,1].imshow(hr[3,1,:,:])
        ax[0,2].imshow(hr[3,2,:,:])
        ax[1,0].imshow(lr[3,0,:,:])
        ax[1,1].imshow(lr[3,1,:,:])
        ax[1,2].imshow(lr[3,2,:,:])
        fig.savefig("debug/check_snapshot_train.png")
        break
    
    for idx,(lr,hr)  in enumerate(val1_loader):
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(hr[3,0,:,:])
        ax[0,1].imshow(hr[3,1,:,:])
        ax[0,2].imshow(hr[3,2,:,:])
        ax[1,0].imshow(lr[3,0,:,:])
        ax[1,1].imshow(lr[3,1,:,:])
        ax[1,2].imshow(lr[3,2,:,:])
        fig.savefig("debug/check_snapshot_val1.png")
        break
    
    for idx,(lr,hr)  in enumerate(val2_loader):
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(hr[3,0,:,:])
        ax[0,1].imshow(hr[3,1,:,:])
        ax[0,2].imshow(hr[3,2,:,:])
        ax[1,0].imshow(lr[3,0,:,:])
        ax[1,1].imshow(lr[3,1,:,:])
        ax[1,2].imshow(lr[3,2,:,:])
        fig.savefig("debug/check_snapshot_val2.png")
        break
        
    for idx,(lr,hr)  in enumerate(test1_loader):
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(hr[3,0,:,:])
        ax[0,1].imshow(hr[3,1,:,:])
        ax[0,2].imshow(hr[3,2,:,:])
        ax[1,0].imshow(lr[3,0,:,:])
        ax[1,1].imshow(lr[3,1,:,:])
        ax[1,2].imshow(lr[3,2,:,:])
        fig.savefig("debug/check_snapshot_test1.png")
        break
    for idx,(lr,hr) in enumerate(test2_loader):
        fig,ax = plt.subplots(2,3)
        ax[0,0].imshow(hr[3,0,:,:])
        ax[0,1].imshow(hr[3,1,:,:])
        ax[0,2].imshow(hr[3,2,:,:])
        ax[1,0].imshow(lr[3,0,:,:])
        ax[1,1].imshow(lr[3,1,:,:])
        ax[1,2].imshow(lr[3,2,:,:])
        fig.savefig("debug/check_snapshot_test2.png")
        break
if __name__ =='__main__':
    main()
