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
id = random.randint(0,10000)
run = neptune.init_run(
    project="junyiICSI/superbenchRebuttal",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGIxYjI4YS0yNDljLTQwOWMtOWY4YS0wOGNhM2Q5Y2RlYzQifQ==",
    tags = [str(id)],
)
# train the model with the given parameters and save the model with the best validation error
def train(args, train_loader, val1_loader, val2_loader, model, optimizer, criterion):
    best_val = np.inf
    train_loss_list, val_error_list = [], []
    start2 = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        train_loss_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # [b,c,h,w]
            data, target = data.float().to(args.device), target.float().to(args.device)

            # forward
            model.train()
            output = model(data) 
            loss = criterion(output, target)
            train_loss_total += loss.item()
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        # record train loss
        train_loss_mean = train_loss_total / len(train_loader)
        train_loss_list.append(train_loss_mean)
        run["train/loss"].log(train_loss_mean)
        # validate
        mse1, mse2 = validate(args, val1_loader, val2_loader, model, criterion)
        print("epoch: %s, val1 error (interp): %.10f, val2 error (extrap): %.10f" % (epoch, mse1, mse2))      
        val_error_list.append(mse1+mse2)
        run["val/loss"].log((mse1+mse2)/2)
        if (mse1+mse2) <= best_val:
            best_val = mse1+mse2
            save_checkpoint(model, optimizer,'results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.noise_ratio) + '_' + str(args.seed) +'_' +str(id) + '.pt')
        end = time.time()
        print('The epoch time is: ', (end - start))
    end2 = time.time()
    print('The training time is: ', (end2 - start2))

    return train_loss_list, val_error_list

# validate the model 
def validate(args, val1_loader, val2_loader, model, criterion):
    mse1 = 0
    mse2 = 0
    rfne_mean =0
    rfne_mean2 =0
    c = 0
    d = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val1_loader):
            data, target = data.float().to(args.device), target.float().to(args.device)
            output = model(data) 
            mse1 += criterion(output, target) * data.shape[0]
            rfne = torch.norm(output - target, p=2, dim=(-1,-2,-3)) / torch.norm(target, p=2, dim=(-1,-2,-3))
            rfne_mean += rfne.mean()
            c += data.shape[0]
            d +=1
    run["val1_rfne"].log(rfne_mean.item()/d)
    mse1 /= c
    c = 0
    d = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val2_loader):
            data, target = data.float().to(args.device), target.float().to(args.device)
            output = model(data) 
            mse2 += criterion(output, target) * data.shape[0]
            rfne = torch.norm(output - target, p=2, dim=(-1,-2,-3)) / torch.norm(target, p=2, dim=(-1,-2,-3))
            rfne_mean2 += rfne.mean()
            c += data.shape[0]
            d +=1
    mse2 /= c
    run["val2_rfne"].log(rfne_mean2.item()/d)
    return mse1.item(), mse2.item()


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
    run["config"] = vars(args)
    # % --- %
    # Set random seed to reproduce the work
    # % --- %
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.save({"config":vars(args),
                "saved_path": str('results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.noise_ratio) + '_' + str(args.seed) +'_' +str(id) + '.pt')},f"results/config_{str(id)}.pt")
    # % --- %
    # Load data
    # % --- %
    resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name) # 
    train_loader, val1_loader, val2_loader, _, _ = getData(args, args.n_patches, std=std)
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
    print(height, width)
    model_list = {
            'subpixelCNN': subpixelCNN(args.in_channels, upscale_factor=args.upscale_factor, width=1, mean = mean,std = std),
            'SRCNN': SRCNN(args.in_channels, args.upscale_factor,mean,std),
            'EDSR': EDSR(args.in_channels, 64, 16, args.upscale_factor, mean, std),
            'WDSR': WDSR(args.in_channels, args.in_channels, 32, 18, args.upscale_factor, mean, std),
            'SwinIR': SwinIR(upscale=args.upscale_factor, in_chans=args.in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
    }

    model = model_list[args.model].to(args.device)
    model = torch.nn.DataParallel(model)
    
    # if pretrain and posttune
    if args.pretrained == True:
        model = load_checkpoint(model, args.model_path)
        model = model.to(args.device)

    # Model summary
    print(model)    
    print('**** Setup ****')
    print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')    

    # % --- %
    # Set optimizer, loss function and Learning Rate Scheduler
    # % --- %
    optimizer = set_optimizer(args, model)
    if args.pretrained == True:
        checkpoint = torch.load(args.model_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer = optimizer.to(args.device)
    scheduler = set_scheduler(args, optimizer, train_loader)
    criterion = loss_function(args)

    # % --- %
    # Training and validation
    # % --- %
    train_loss_list, val_error_list = train(args, train_loader, val1_loader, val2_loader, model, optimizer, criterion)

    # % --- %
    # Post-process: plot train loss and val error
    # % --- %
    x_axis = np.arange(0, args.epochs)
    plt.figure()
    plt.plot(x_axis, train_loss_list, label = 'train loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./figures/train_loss_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.seed) + '.png', dpi = 300)

    plt.figure()
    plt.plot(x_axis, val_error_list, label = 'val error')
    plt.yscale('log')
    plt.legend()
    plt.savefig('./figures/val_error_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.seed) + '.png', dpi = 300)

if __name__ =='__main__':
    main()
