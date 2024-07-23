'''training function'''

import numpy as np
import torch
from torch import nn
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from src.models import *
from data_loader import getData
from utils import *
import neptune
id = torch.randint(1000,(1,1))
run = neptune.init_run(
    project="junyiICSI/superbenchRebuttal",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGIxYjI4YS0yNDljLTQwOWMtOWY4YS0wOGNhM2Q5Y2RlYzQifQ==",
    tags = [str(id.item())],
)
# train the model with the given parameters and save the model with the best validation error
def train(args, train_loader, val1_loader, val2_loader, model, optimizer ,scheduler,criterion):
    best_val = np.inf
    train_loss_list, val_error_list = [], []
    start2 = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        train_loss_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # [b,c,h,w]
            data, target = data.to(args.device).float(), target.to(args.device).float()
            # forward
            model.train()
            output = model(data) 
            loss = criterion(output, target)
            train_loss_total += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        run["train/lr"].log(scheduler.get_last_lr()[0])
        # record train loss
        train_loss_mean = train_loss_total / len(train_loader)
        train_loss_list.append(train_loss_mean)
        run["train/loss"].log(train_loss_mean)
        # validate
        mse1, mse2 = validate(args, val1_loader, val2_loader, model, criterion)
        run["val/val1_error"].log(mse1)
        run["val/val2_error"].log(mse2)
        print("epoch: %s, val1 error (interp): %.10f, val2 error (extrap): %.10f" % (epoch, mse1, mse2))      
        val_error_list.append(mse1+mse2)

        if (mse1+mse2) <= best_val:
            best_val = mse1+mse2
            save_checkpoint(model, optimizer,'results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.noise_ratio) + '_' + str(args.seed) +'_' +str(id.item()) + '.pt')
        end = time.time()
        print('The epoch time is: ', (end - start))
        # if epoch % 50 ==0:
        #     RFNE1,RFNE2 = quick_RFNE(args,val1_loader,val2_loader,model)
        #     run["val/RFNE1"].log(RFNE1)
        #     run["val/RFNE2"].log(RFNE2)
        #     print("epoch: %s, val1 RFNE (interp): %.10f, val2 RFNE (extrap): %.10f" % (epoch, RFNE1, RFNE2))
    end2 = time.time()
    print('The training time is: ', (end2 - start2))

    return train_loss_list, val_error_list

# validate the model 
def validate(args, val1_loader, val2_loader, model, criterion):
    mse1 = 0
    mse2 = 0
    c = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val1_loader):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            mse1 += criterion(output, target) * data.shape[0]
            c += data.shape[0]
    mse1 /= c
    c = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val2_loader):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            mse2 += criterion(output, target) * data.shape[0]
            c += data.shape[0]
    mse2 /= c

    return mse1.item(), mse2.item()

def quick_RFNE(args,test1_loader,test2_loader,model,mean,std):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test1_loader):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            RFNE1 = torch.norm(output- target, p =2,dim =(-1,-2))/torch.norm(target, p =2,dim =(-1,-2))
            err1 += RFNE1.mean().item()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test2_loader):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            RFNE2 = torch.norm(output- target, p =2,dim =(-1,-2))/torch.norm(target, p =2,dim =(-1,-2))
    return RFNE1.mean().item(),RFNE2.mean().item()

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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
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
    parser.add_argument('--modes', type=int, default=12, help='num of modes in first dimension')
    parser.add_argument('--loss_type', type=str, default='l2', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='StepLR', help='type of scheduler')

    args = parser.parse_args()
    print(args)
    run['config'] = vars(args)

    # % --- %
    # Set random seed to reproduce the work
    # % --- %
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # % --- %
    # Load data
    # % --- %
    resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name) # 
    train_loader, val1_loader, val2_loader,TEST1, TEST2= getData(args, args.n_patches, std=std)
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

    model = model_list[args.model].to(args.device)    
    # if pretrain and posttune
    if args.pretrained == True:
        print('Loading pretrained model...')
        optimizer = set_optimizer(args, model)
        model,optimizer = load_checkpoint(model,optimizer, args.model_path)
        model = model.to(args.device)
        criterion = loss_function(args)
        scheduler = set_scheduler(args, optimizer, train_loader)
    else:
        optimizer = set_optimizer(args, model) # weight decay for adam must be 0, idk why 
        scheduler = set_scheduler(args, optimizer, train_loader)
        criterion = loss_function(args)

    # Model summary
    print(model)    
    print('**** Setup ****')
    print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')    

    # % --- %
    # Set optimizer, loss function and Learning Rate Scheduler
    # % --- %

    # % --- %
    # Training and validation
    # % --- %

    train_loss_list, val_error_list = train(args, train_loader, val1_loader, val2_loader, model, optimizer,scheduler,criterion)

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
