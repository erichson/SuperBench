'''Evaluation function'''

import numpy as np
import torch
from torch import nn
import argparse
import matplotlib.pyplot as plt
import cmocean  
import math
import torch.nn.functional as F
from data_loader import getData
from utils import *
from src.models import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os 


DIR = "/pscratch/sd/j/junyi012/superbench_v2/"
def load_everything_uniform(args, test1_loader, test2_loader, model, mean, std,location=(3,0)):
    '''load any model and return LR and HR to buffer'''
    if args.model != 'FNO2D_patch':
        with torch.no_grad():
            lr_list, hr_list, pred_list = [], [], []
            for batch_idx, (data, target) in enumerate(test2_loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data)
                lr,hr,pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
                lr_list.append(lr)
                hr_list.append(hr)
                pred_list.append(pred)
            pred_list = np.concatenate(pred_list)
            lr_list = np.concatenate(lr_list)
            hr_list = np.concatenate(hr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_{args.noise_ratio}_lr_uniform.npy",lr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_{args.noise_ratio}_hr_uniform.npy",hr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_{args.model}_{args.noise_ratio}_pred_uniform.npy",pred_list)
    else:
        raise NotImplementedError
        # with torch.no_grad():
        #     lr_list, hr_list, pred_list = [], [], []
        #     for batch_idx, (data, target) in enumerate(test2_loader):
        #         data, target = data.to(args.device).float(), target.to(args.device).float()
        #         hr_patch_size = 128
        #         hr_stride = 128
        #         lr_patch_size = 128//args.upscale_factor
        #         lr_stride = 128//args.upscale_factor
        #         lr_patches = data.unfold(2, lr_patch_size, lr_stride).unfold(3, lr_patch_size, lr_stride)
        #         hr_patches = target.unfold(2, hr_patch_size, hr_stride).unfold(3, hr_patch_size, hr_stride)
        #         if lr_patches.shape[2] != hr_patches.shape[2] or lr_patches.shape[3] != hr_patches.shape[3]:
        #             print("patch size not match")
        #             return False
        #         output = torch.zeros_like(hr_patches)
        #         for i in range(hr_patches.shape[2]):
        #             for j in range(hr_patches.shape[3]):
        #                 lr = lr_patches[:,:,i,j]
        #                 with torch.no_grad():
        #                     output_p = model(lr)
        #                     output[:,:,i,j] = output_p
        #         patches_flat = output.permute(0, 1, 4, 5, 2, 3).contiguous().view(1, target.shape[1]*hr_patch_size**2, -1)
        #         output = F.fold(patches_flat, output_size=(target.shape[-2], target.shape[-1]), kernel_size=(hr_patch_size, hr_patch_size), stride=(hr_stride, hr_stride))
        #         lr_data,hr_data,pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
        #         lr_list.append(lr_data)
        #         hr_list.append(hr_data)
        #         pred_list.append(pred)
        #     pred_list = np.concatenate(pred_list)
        #     lr_list = np.concatenate(lr_list)
        #     hr_list = np.concatenate(hr_list)
        #     # if os.path.exists(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_lr.npy") == False:

    return True

def load_everything(args, test1_loader, test2_loader, model, mean, std,location=(3,0)):
    '''load any model and return LR and HR to buffer'''
    if args.model != 'FNO2D_patch':
        with torch.no_grad():
            lr_list, hr_list, pred_list = [], [], []
            for batch_idx, (data, target) in enumerate(test2_loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data)
                lr,hr,pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
                lr_list.append(lr)
                hr_list.append(hr)
                pred_list.append(pred)
            pred_list = np.concatenate(pred_list)
            lr_list = np.concatenate(lr_list)
            hr_list = np.concatenate(hr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_lr.npy",lr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_hr.npy",hr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_{args.model}_pred.npy",pred_list)
    else:
        with torch.no_grad():
            lr_list, hr_list, pred_list = [], [], []
            for batch_idx, (data, target) in enumerate(test2_loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                hr_patch_size = 128
                hr_stride = 128
                lr_patch_size = 128//args.upscale_factor
                lr_stride = 128//args.upscale_factor
                lr_patches = data.unfold(2, lr_patch_size, lr_stride).unfold(3, lr_patch_size, lr_stride)
                hr_patches = target.unfold(2, hr_patch_size, hr_stride).unfold(3, hr_patch_size, hr_stride)
                if lr_patches.shape[2] != hr_patches.shape[2] or lr_patches.shape[3] != hr_patches.shape[3]:
                    print("patch size not match")
                    return False
                output = torch.zeros_like(hr_patches)
                for i in range(hr_patches.shape[2]):
                    for j in range(hr_patches.shape[3]):
                        lr = lr_patches[:,:,i,j]
                        with torch.no_grad():
                            output_p = model(lr)
                            output[:,:,i,j] = output_p
                patches_flat = output.permute(0, 1, 4, 5, 2, 3).contiguous().view(1, target.shape[1]*hr_patch_size**2, -1)
                output = F.fold(patches_flat, output_size=(target.shape[-2], target.shape[-1]), kernel_size=(hr_patch_size, hr_patch_size), stride=(hr_stride, hr_stride))
                lr_data,hr_data,pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
                lr_list.append(lr_data)
                hr_list.append(hr_data)
                pred_list.append(pred)
            pred_list = np.concatenate(pred_list)
            lr_list = np.concatenate(lr_list)
            hr_list = np.concatenate(hr_list)
            # if os.path.exists(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_lr.npy") == False:
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_lr.npy",lr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_hr.npy",hr_list)
            np.save(DIR+f"eval_buffer/{args.data_name}_{args.upscale_factor}_{args.model}_pred.npy",pred_list)
    return True


def load_lrhr(args, test1_loader, test2_loader, model, mean, std,location=(3,0)):
    if os.path.exists(DIR+f"plot_buffer/{args.data_name}_{args.upscale_factor}_lr.npy") == False:
        '''load any model and return LR and HR to buffer'''
        with torch.no_grad():
            rfne_list =[]
            lr_list, hr_list, pred_list = [], [], []
            for batch_idx, (data, target) in enumerate(test2_loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data)
                rfne = torch.norm((target-output),p =2,dim=(-1,-2)) / torch.norm(target,p =2,dim=(-1,-2))
                rfne = rfne[:,-1]
                rfne_list.append(rfne.cpu().numpy())
                lr,hr,pred = data.cpu().numpy(), target.cpu().numpy(), output.cpu().numpy()
                lr_list.append(lr)
                hr_list.append(hr)
                pred_list.append(pred)
            rfne_list = np.concatenate(rfne_list)
            lr_list = np.concatenate(lr_list)
            hr_list = np.concatenate(hr_list)
            np.save(DIR+f"plot_buffer/{args.data_name}_{args.upscale_factor}_lr.npy",lr_list)
            np.save(DIR+f"plot_buffer/{args.data_name}_{args.upscale_factor}_hr.npy",hr_list)
            np.save(DIR+f"plot_buffer/{args.data_name}_{args.upscale_factor}_rfne.npy",rfne_list)
            print(lr_list.shape, hr_list.shape)
    else:
        lr_list = np.load(DIR+f"plot_buffer/{args.data_name}_{args.upscale_factor}_lr.npy")
        hr_list = np.load(DIR+f"plot_buffer/{args.data_name}_{args.upscale_factor}_hr.npy")
    return lr_list, hr_list

def get_pred(args, lr, hr, model, mean, std,location=(3,0)):
    if args.model != 'FNO2D_patch':
        batch, channel = location 
        data, target = lr[batch:batch+1], hr[batch:batch+1]
        save_name = DIR+f"plot_buffer/{args.data_name}_{args.model}_{args.upscale_factor}_pred_b{batch}c{channel}.npy"
        data, target = torch.from_numpy(data).to(args.device).float(), torch.from_numpy(target).to(args.device).float()
        with torch.no_grad():
            output = model(data)
            output = output.cpu().numpy()
            if os.path.exists(save_name) == True:
                np.save(save_name,output)
            else:
                print("pred has been saved")
    else:
        batch, channel = location 
        data, target = lr[batch:batch+1], hr[batch:batch+1]

        save_name = DIR+f"plot_buffer/{args.data_name}_{args.model}_{args.upscale_factor}_pred_b{batch}c{channel}.npy"
        with torch.no_grad():
            data, target = torch.from_numpy(data).to(args.device).float(), torch.from_numpy(target).to(args.device).float()
            hr_patch_size = 128
            hr_stride = 128
            lr_patch_size = 128//args.upscale_factor
            lr_stride = 128//args.upscale_factor
            lr_patches = data.unfold(2, lr_patch_size, lr_stride).unfold(3, lr_patch_size, lr_stride)
            hr_patches = target.unfold(2, hr_patch_size, hr_stride).unfold(3, hr_patch_size, hr_stride)
            if lr_patches.shape[2] != hr_patches.shape[2] or lr_patches.shape[3] != hr_patches.shape[3]:
                print("patch size not match")
                return False
            output = torch.zeros_like(hr_patches)
            for i in range(hr_patches.shape[2]):
                for j in range(hr_patches.shape[3]):
                    lr = lr_patches[:,:,i,j]
                    with torch.no_grad():
                        output_p = model(lr)
                        output[:,:,i,j] = output_p
            patches_flat = output.permute(0, 1, 4, 5, 2, 3).contiguous().view(1, hr.shape[1]*hr_patch_size**2, -1)
            # Fold the patches back
            import torch.nn.functional as F
            output = F.fold(patches_flat, output_size=(hr.shape[-2], hr.shape[-1]), kernel_size=(hr_patch_size, hr_patch_size), stride=(hr_stride, hr_stride))
        output = output.cpu().numpy()
        # if os.path.exists(save_name) == False:
        np.save(save_name,output)
        # else:
        #     print("pred has been saved")

    return True

def main():  
    parser = argparse.ArgumentParser(description='training parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='nskt_16k', help='dataset')
    parser.add_argument('--data_path', type=str, default='./datasets/nskt16000_1024', help='the folder path of dataset')
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size for high-resolution snapshots')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')

    # arguments for evaluation
    parser.add_argument('--model', type=str, default='shallowDecoder', help='model')
    parser.add_argument('--model_path', type=str, default=None, help='saved model')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')
    
    # arguments for training
    parser.add_argument('--epochs', type=int, default=300, help='max epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--step_size', type=int, default=100, help='step size for scheduler')
    parser.add_argument('--gamma', type=float, default=0.97, help='coefficient for scheduler')

    # arguments for model
    parser.add_argument('--loss_type', type=str, default='l1', help='L1 or L2 loss')
    parser.add_argument('--optimizer_type', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--scheduler_type', type=str, default='ExponentialLR', help='type of scheduler')
    parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
    parser.add_argument('--in_channels', type=int, default=1, help='num of input channels')
    parser.add_argument('--hidden_channels', type=int, default=32, help='num of hidden channels')
    parser.add_argument('--out_channels', type=int, default=1, help='num of output channels')
    parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')

    args = parser.parse_args()
    print(args)

    # % --- %
    # Set random seed to reproduce the work
    # % --- %
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # % --- %
    # Load data
    # % --- %
    resol, n_fields, n_train_samples, mean, std = get_data_info(args.data_name)
    test1_loader, test2_loader = getData(args, args.n_patches, std=std,test=True)

    # % --- %
    # Get model
    # % --- %
    upscale = args.upscale_factor
    window_size = 8
    height = (args.crop_size // upscale // window_size + 1) * window_size
    width = (args.crop_size // upscale // window_size + 1) * window_size
    if args.data_name == 'era5':
        height = (720 // upscale // window_size + 1) * window_size # for era5 
        width = (1440 // upscale // window_size + 1) * window_size # for era5
    model_list = {
            'subpixelCNN': subpixelCNN(args.in_channels, upscale_factor=args.upscale_factor, width=1, mean = mean,std = std),
            'SRCNN': SRCNN(args.in_channels, args.upscale_factor,mean,std),
            'EDSR': EDSR(args.in_channels, 64, 16, args.upscale_factor, mean, std),
            'WDSR': WDSR(args.in_channels,args.in_channels, 32,18, args.upscale_factor, mean, std),
            'SwinIR': SwinIR(upscale=args.upscale_factor, in_chans=args.in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
            'Bicubic': Bicubic(args.upscale_factor),
    }
    # Regarding train with physics loss
    if args.model.startswith('SwinIR'):
        name = "SwinIR"
    else:
        name = args.model

    model = model_list[name]
    model = torch.nn.DataParallel(model)
    if args.model_path is None:
        model_path = 'results/model_' + str(args.model) + '_' + str(args.data_name) + '_' + str(args.upscale_factor) + '_' + str(args.lr) + '_' + str(args.method) +'_' + str(args.noise_ratio) + '_' + str(args.seed) + '.pt'
    else:
        model_path = args.model_path
    if args.model != 'Bicubic':
        model = load_checkpoint(model, model_path)
        model = model.to(args.device)

        # Model summary   
        print('**** Setup ****')
        print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')    

    else: 
        print('Using bicubic interpolation...')  
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
    # load_everything_uniform(args, test1_loader, test2_loader, model, mean, std)
if __name__ =='__main__':
    main()