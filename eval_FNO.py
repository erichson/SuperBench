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

# % --- %
# Evaluate models
# % --- %
class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) // 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol    

class LossGenerator(nn.Module):
    def __init__(self, args, dx=2.0*math.pi/2048.0, kernel_size=3):
        super(LossGenerator,self).__init__()

        self.delta_x = torch.tensor(dx)

        #https://en.wikipedia.org/wiki/Finite_difference_coefficient
        self.filter_y4 = [[[[    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0],
           [1/12, -8/12,  0,  8/12, -1/12],
           [    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0]]]]

        self.filter_x4 = [[[[    0,   0,   1/12,   0,     0],
           [    0,   0,   -8/12,   0,     0],
           [    0,   0,   0,   0,     0],
           [    0,   0,   8/12,   0,     0],
           [    0,   0,   -1/12,   0,     0]]]]

        self.filter_x2 = [[[[    0,   -1/2,   0],
                    [    0,   0,   0],
                    [     0,   1/2,   0]]]]

        self.filter_y2 = [[[[    0,   0,   0],
                    [    -1/2,   0,   1/2],
                    [     0,   0,   0]]]]

        if kernel_size ==5:
            self.dx = Conv2dDerivative(
                DerFilter = self.filter_x4,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'dx_operator').to(args.device)

            self.dy = Conv2dDerivative(
                DerFilter = self.filter_y4,
                resol = self.delta_x,
                kernel_size = 5,
                name = 'dy_operator').to(args.device)  

        elif kernel_size ==3:
            self.dx = Conv2dDerivative(
                DerFilter = self.filter_x2,
                resol = self.delta_x,
                kernel_size = 3,
                name = 'dx_operator').to(args.device)

            self.dy = Conv2dDerivative(
                DerFilter = self.filter_y2,
                resol = self.delta_x,
                kernel_size = 3,
                name = 'dy_operator').to(args.device)  

    def get_div_loss(self, output):
        '''compute divergence loss'''
        u = output[:,0:1,:,:]
        #bu,xu,yu = u.shape
        #u = u.reshape(bu,1,xu,yu)

        v = output[:,1:2,:,:]
        #bv,xv,yv = v.shape
        #v = v.reshape(bv,1,xv,yv)

        #w = output[:,0,:,:]
        u_x = self.dx(u)  
        v_y = self.dy(v)  
        # div
        div = u_x + v_y

        return div

def validate_phyLoss(args, test1_loader, test2_loader, model):
    
    avg_phyloss1, avg_phyloss2 = 0.0, 0.0

    MSEfunc = nn.MSELoss()
    lossgen = LossGenerator(args, dx=2.0*math.pi/2048.0, kernel_size=5)
    
    c = 0
    with torch.no_grad():
        for batch in test1_loader:
            input, target = batch[0].float().to(args.device), batch[1].float().to(args.device)
            model.eval()
            out = model(input)
            div = lossgen.get_div_loss(output=out)
            phy_loss = MSEfunc(div, torch.zeros_like(div).to(args.device)) # calculating physics loss
            avg_phyloss1 += phy_loss.item() * target.shape[0]
            c += target.shape[0]
    avg_phyloss1 /= c

    c = 0
    with torch.no_grad():
        for batch in test2_loader:
            input, target = batch[0].float().to(args.device), batch[1].float().to(args.device)
            model.eval()
            out = model(input)
            div = lossgen.get_div_loss(output=out)
            phy_loss = MSEfunc(div, torch.zeros_like(div).to(args.device)) # calculating physics loss
            avg_phyloss2 += phy_loss.item() * target.shape[0]
            c += target.shape[0]
    avg_phyloss2 /= c

    return avg_phyloss1, avg_phyloss2

def normalize(args,target,mean,std):
    mean = torch.Tensor(mean).to(args.device).view(1,target.shape[1],1,1)
    std = torch.Tensor(std).to(args.device).view(1,target.shape[1],1,1)
    target = (target - mean) / std
    return target
def validate_all_metrics(args, test1_loader, test2_loader, model, mean, std):
    from torchmetrics import StructuralSimilarityIndexMeasure

    ssim = StructuralSimilarityIndexMeasure().to(args.device)
    rine1, rine2, rfne1, rfne2, psnr1, psnr2, ssim1, ssim2,mse1,mse2,mae1,mae2 = [], [], [], [], [], [], [], [],[],[],[],[]

    # Helper function for PSNR
    def compute_psnr(true, pred):
        mse = torch.mean((true - pred) ** 2)
        if mse == 0:
            raise ValueError("Mean squared error is zero, cannot compute PSNR.")
            # mse = 1e-3
        max_value = torch.max(true)
        psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
        if np.isnan(psnr.cpu().numpy()) == True:
            output= torch.zeros_like(psnr)
        else:
            output = psnr
        return output

    with torch.no_grad():
        for loader, (rine_list, rfne_list, psnr_list, ssim_list,mse_list,mae_list) in zip([test1_loader, test2_loader],
                                                                        [(rine1, rfne1, psnr1, ssim1,mse1,mae1),
                                                                         (rine2, rfne2, psnr2, ssim2,mse2,mae2)]):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data)
                output = normalize(args, output, mean, std)
                target = normalize(args, target, mean, std)

                # MSE 
                mse = torch.mean((target - output) ** 2,dim =(-1,-2,-3))
                mse_list.append(mse.mean())

                # MAE
                mae = torch.mean(torch.abs(target - output),dim=(-1,-2,-3))
                mae_list.append(mae.mean())
                # RINE
                err_ine = torch.norm(target-output, p=np.inf, dim=(-1, -2)) / torch.norm(target, p=np.inf, dim=(-1, -2))
                rine_list.append(err_ine.mean())

                # RFNE
                err_rfne = torch.norm(target-output, p=2, dim=(-1, -2)) / torch.norm(target, p=2, dim=(-1, -2))
                rfne_list.append(err_rfne.mean())

                # PSNR
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        err_psnr = compute_psnr(target[i, j, ...], output[i, j, ...])
                        psnr_list.append(err_psnr)

                # SSIM
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        err_ssim = ssim(target[i:(i+1), j:(j+1), ...], output[i:(i+1), j:(j+1), ...])
                        ssim_list.append(err_ssim.cpu())

    # Averaging and converting to scalar values
    avg_rine1, avg_rine2 = torch.mean(torch.stack(rine1)).item(), torch.mean(torch.stack(rine2)).item()
    avg_rfne1, avg_rfne2 = torch.mean(torch.stack(rfne1)).item(), torch.mean(torch.stack(rfne2)).item()
    avg_psnr1, avg_psnr2 = torch.mean(torch.stack(psnr1)).item(), torch.mean(torch.stack(psnr2)).item()
    avg_ssim1, avg_ssim2 = torch.mean(torch.stack(ssim1)).item(), torch.mean(torch.stack(ssim2)).item()
    avg_mse1,avg_mse2 = torch.mean(torch.stack(mse1)).item(), torch.mean(torch.stack(mse2)).item()
    avg_mae1,avg_mae2 = torch.mean(torch.stack(mae1)).item(), torch.mean(torch.stack(mae2)).item()
    return (avg_rine1, avg_rine2), (avg_rfne1, avg_rfne2), (avg_psnr1, avg_psnr2), (avg_ssim1, avg_ssim2),(avg_mse1,avg_mse2),(avg_mae1,avg_mae2)
    
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
    train_loader, val1_loader, val2_loader,test1_loader, test2_loader= getData(args, args.n_patches, std=std,patched_eval=flag)
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


    import json
    
    # Check if the results file already exists and load it, otherwise initialize an empty list
    try:
        with open("normed_eval.json", "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
        print("No results file found, initializing a new one.")
    # Create a unique key based on your parameters
    key = f"{args.model}_{args.data_name}_{args.method}_{args.upscale_factor}_{args.noise_ratio}"

# Check if the key already exists in the dictionary
    if key not in all_results:
        all_results[key] = {
            "model": args.model,
            "dataset": args.data_name,
            "method": args.method,
            "scale factor": args.upscale_factor,
            "noise ratio": args.noise_ratio,
            "parameters": (sum(p.numel() for p in model.parameters())/1000000.0),
            "metrics": {}
        }

    INE, RFNE, PSNR, SSIM,MSE,MAE = validate_all_metrics(args, test1_loader, test2_loader, model, mean, std)
    # Validate and store Infinity norm results
    # ine1, ine2 = validate_RINE(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["IN"] = {'test1 error': INE[0], 'test2 error': INE[1]}

    # Validate and store RFNE results
    # error1, error2 = validate_RFNE(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["RFNE"] = {'test1 error': RFNE[0], 'test2 error': RFNE[1]}

    # Validate and store PSNR results
    # error1, error2 = validate_PSNR(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["PSNR"] = {'test1 error': PSNR[0], 'test2 error': PSNR[1]}

    # Validate and store SSIM results
    # error1, error2 = validate_SSIM(args, test1_loader, test2_loader, model, mean, std)
    all_results[key]["metrics"]["SSIM"] = {'test1 error': SSIM[0], 'test2 error': SSIM[1]}
    # Validate and store MSE results
    all_results[key]["metrics"]["MSE"] = {'test1 error': MSE[0], 'test2 error': MSE[1]}
    # Validate and store MAE results
    all_results[key]["metrics"]["MAE"] = {'test1 error': MAE[0], 'test2 error': MAE[1]}
    # Validate and store Physics loss results for specific data names
    if args.data_name in ["nskt_16k", "nskt_32k","nskt_16k_sim","nskt_32k_sim"] or args.data_name.startswith("nskt"):
        phy_err1, phy_err2 = validate_phyLoss(args, test1_loader, test2_loader, model)
        all_results[key]["metrics"]["Physics"] = {'test1 error': phy_err1, 'test2 error': phy_err2}

    # Serialize the updated results list to the JSON file
    with open("normed_eval.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ =='__main__':
    main()