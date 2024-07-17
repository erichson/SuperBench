'''Evaluation function'''

import numpy as np
import torch
from torch import nn
import argparse
import matplotlib.pyplot as plt
import cmocean  
import math
from data_loader import getData
from utils import *
from src.models import *
from eval_plot import load_lrhr
from eval_plot import get_pred
from eval_plot import load_everything
def main():  
    parser = argparse.ArgumentParser(description='training parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='nskt_16k', help='dataset')
    parser.add_argument('--data_path', type=str, default='../superbench/datasets/nskt16000_1024', help='the folder path of dataset')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size for high-resolution snapshots')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')    
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'), help='load the pretrained model')
    
    # arguments for training
    parser.add_argument('--model', type=str, default='FNO2D', help='model')
    parser.add_argument('--epochs', type=int, default=500, help='max epochs')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--step_size', type=int, default=50, help='step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='coefficient for scheduler')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

    # arguments for model
    parser.add_argument('--upscale_factor', type=int, default=8, help='upscale factor')
    parser.add_argument('--in_channels', type=int, default=3, help='num of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='num of output channels')
    parser.add_argument('--hidden_channels', type=int, default=64, help='num of output channels')
    parser.add_argument('--modes', type=int, default=20, help='num of modes in first dimension')
    parser.add_argument('--loss_type', type=str, default='l2', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='StepLR', help='type of scheduler')

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
    if args.model == "FNO2D_patch":
        flag = True
    else :
        flag = False

    resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name) # 
    train_loader, val1_loader, val2_loader,test1_loader, test2_loader= getData(args, args.n_patches, std=std,patched_eval=False)
    print('The data resolution is: ', resol)
    print("mean is: ",mean)
    print("std is: ",std)

    # % --- %
    # Get model
    # % --- %
    # some hyper-parameters for SwinIR
    upscale = args.upscale_factor
    hidden = args.hidden_channels
    modes = args.modes
    model_list = {  
        "FNO2D":FNO2D(layers=[hidden, hidden, hidden, hidden, hidden],modes1=[modes, modes, modes, modes],modes2=[modes, modes, modes, modes],fc_dim=128,in_dim=args.in_channels,out_dim=args.in_channels,mean= mean,std=std,scale_factor=upscale),
        "FNO2D_conv":FNO2D_conv(layers=[hidden, hidden, hidden, hidden, hidden],modes1=[modes, modes, modes, modes],modes2=[modes, modes, modes, modes],fc_dim=128,in_dim=args.in_channels,out_dim=args.in_channels,mean= mean,std=std,scale_factor=upscale),
    }

    if args.model == "FNO2D_patch":
        model = model_list["FNO2D"].to(args.device)    
    else:
        model = model_list[args.model].to(args.device)
    # if pretrain and posttune
    model= load_checkpoint(model, args.model_path)

    model = model.to(args.device)

        # Model summary   
    print('**** Setup ****')
    print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')    
    
    # lr,hr = load_lrhr(args, test1_loader, test2_loader, model, mean, std)
    # if args.data_name == 'nskt_16k':
    #     batch, channel = 16, -1
    # elif args.data_name == "nskt_32k":
    #     batch, channel = 16, -1
    # elif args.data_name == "era5":
    #     batch, channel = 55, 0
    # elif args.data_name == "cosmo":
    #     batch, channel = 55, 0
    # elif args.data_name == "nskt_32k_sim_4_v8":
    #     batch,channel = 12,-1
    # elif args.data_name == "nskt_16k_sim_4_v8":
    #     batch,channel = 12,-1
    # elif args.data_name =="cosmo_sim_8":
    #     batch,channel = 55,0
    # get_pred(args, lr, hr, model, mean, std,location=(batch,channel))
    load_everything(args, test1_loader, test2_loader, model, mean, std)

if __name__ =='__main__':
    main()