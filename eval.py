'''Evaluation function'''

import numpy as np
import torch
from torch import nn
import argparse
import matplotlib.pyplot as plt
import cmocean  
import math
from src.data_loader_crop import getData
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

def validate_RINE(args, test1_loader, test2_loader, model,mean,std):
    '''Relative infinity norm error (RINE)'''
    # calculate the RINE of each snapshot and then average
    ine1 = [] 
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float() # [b,c,h,w]
            output = model(data) 
            output = normalize(args,output,mean,std)
            target = normalize(args,target,mean,std)
            # calculate infinity norm of each snapshot
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    err_ine = torch.norm((target[i,j,...]-output[i,j,...]), p=np.inf)
                    ine1.append(err_ine)
    ine1 = torch.mean(torch.tensor(ine1)).item()

    ine2 = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test2_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data)  
            # calculate infinity norm of each snapshot
            output = normalize(args,output,mean,std)
            target = normalize(args,target,mean,std)
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    err_ine = torch.norm((target[i,j,...]-output[i,j,...]), p=np.inf)
                    ine2.append(err_ine)
    ine2 = torch.mean(torch.tensor(ine2)).item()    
    return ine1, ine2 


def validate_RFNE(args, test1_loader, test2_loader, model,mean,std):
    '''Relative Frobenius norm error (RFNE)'''
    rfne1 = []   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data)   
            output = normalize(args,output,mean,std)
            target = normalize(args,target,mean,std)
            # calculate frobenius norm of each snapshot of each channel
            for i in range(target.shape[0]):
                for j in range (target.shape[1]):
                    err_rfne = torch.norm((target[i,j,...]-output[i,j,...])) / torch.norm(target[i,j,...])
                    rfne1.append(err_rfne)
    rfne1 = torch.mean(torch.tensor(rfne1)).item()

    rfne2 = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test2_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            output = normalize(args,output,mean,std)
            target = normalize(args,target,mean,std)
            # calculate frobenius norm of each snapshot
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    err_rfne = torch.norm((target[i,j,...]-output[i,j,...])) / torch.norm(target[i,j,...])
                #fne2.append(err_fne)
                    rfne2.append(err_rfne)
    #fne2 = np.array(fne2).mean()
    rfne2 = torch.mean(torch.tensor(rfne2)).item()

    return rfne1, rfne2



def psnr(true, pred):
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float('inf')
    max_value = torch.max(true)
    return 20 * torch.log10(max_value / torch.sqrt(mse))


def validate_PSNR(args, test1_loader, test2_loader, model,mean,std):
    '''Peak signal-to-noise ratio (PSNR)'''
    
    error1 = []   
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test1_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            output = normalize(args,output,mean,std)
            target = normalize(args,target,mean,std)
            # calculate PSNR of each snapshot and then average (Change to channel-wise)
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    err_psnr = psnr(target[i,j,...], output[i,j,...])
                    error1.append(err_psnr)
    error1 = torch.mean(torch.tensor(error1)).item()

    error2 = []  
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate((test2_loader)):
            data, target = data.to(args.device).float(), target.to(args.device).float()
            output = model(data) 
            output = normalize(args,output,mean,std)
            target = normalize(args,target,mean,std)
            # calculate PSNR of each snapshot and then average
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    err_psnr = psnr(target[i,j,...], output[i,j,...])
                    error2.append(err_psnr)
    error2 = torch.mean(torch.tensor(error2)).item()

    return error1, error2


def validate_SSIM(args, test1_loader, test2_loader, model,mean,std):
        '''Structual Similarity Index Measure (SSIM)'''
        from torchmetrics import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure().to(args.device)
        
        error1 = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate((test1_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data) 
                
                output = normalize(args,output,mean,std)
                target = normalize(args,target,mean,std)
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
                        error1.append(err_ssim.cpu())

        # averaged SSIM
        err1 = torch.mean(torch.tensor(error1)).item()

        error2 = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate((test2_loader)):
                data, target = data.to(args.device).float(), target.to(args.device).float()
                output = model(data) 
                output = normalize(args,output,mean,std)
                target = normalize(args,target,mean,std)                
                for i in range(target.shape[0]):
                    for j in range(target.shape[1]):
                        err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
                        error2.append(err_ssim.cpu())

        err2 = torch.mean(torch.tensor(error2)).item()

        return err1, err2

    
def main():  
    parser = argparse.ArgumentParser(description='training parameters')
    # arguments for data
    parser.add_argument('--data_name', type=str, default='nskt_16k', help='dataset')
    parser.add_argument('--data_path', type=str, default='../superbench/datasets/nskt16000_1024', help='the folder path of dataset')
    parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
    parser.add_argument('--crop_size', type=int, default=128, help='crop size for high-resolution snapshots')
    parser.add_argument('--n_patches', type=int, default=8, help='number of patches')

    # arguments for evaluation
    parser.add_argument('--model', type=str, default='shallowDecoder', help='model')
    parser.add_argument('--model_path', type=str, default='results/model_SwinIR_nskt_16k_4_0.0001_bicubic_0.0_5544.pt', help='saved model')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=5544, help='random seed')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')
    parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'), help='load the pretrained model')
    
    # arguments for training
    parser.add_argument('--epochs', type=int, default=300, help='max epochs')
    parser.add_argument('--iterations_per_epoch', type=int, default=1000, help='iterations per epoch')
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
    _, _, _, test1_loader, test2_loader = getData(args, args.n_patches, std=std)

    # % --- %
    # Get model
    # % --- %
    upscale = args.upscale_factor
    window_size = 8
    height = (resol[0] // upscale // window_size + 1) * window_size
    width = (resol[1] // upscale // window_size + 1) * window_size
    model_list = {
            'subpixelCNN': subpixelCNN(args.in_channels, upscale_factor=args.upscale_factor, width=1, mean = mean,std = std),
            'SRCNN': SRCNN(args.in_channels, args.upscale_factor,mean,std),
            'EDSR': EDSR(args.in_channels, args.hidden_channels, args.n_res_blocks, args.upscale_factor, mean, std),
            'WDSR': WDSR(args.in_channels, args.out_channels, args.hidden_channels, args.n_res_blocks, args.upscale_factor, mean, std),
            'SwinIR': SwinIR(upscale=args.upscale_factor, in_chans=args.in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
    }

    model = model_list[args.model]
    model = torch.nn.DataParallel(model)
    
    if args.model != 'bicubic':
        model = load_checkpoint(model, args.model_path)
        model = model.to(args.device)

        # Model summary   
        # print('**** Setup ****')
        # print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        # print('************')    

    else: 
        print('Using bicubic interpolation...')  

    # =============== validate ======================
    with open("result.txt", "a") as f:
        print(" model" + str(args.model) + " data: " + str(args.data_name)+ "  method: " + str(args.method) +" scale factor " + str(args.upscale_factor) + " noise ratio: " + str(args.noise_ratio),file = f)
        ine1, ine2 = validate_RINE(args, test1_loader, test2_loader, model,mean,std)
        print("Infinity norm --- test1 error: %.8f, test2 error: %.8f" % (ine1, ine2),file = f) 

        error1, error2 = validate_RFNE(args, test1_loader, test2_loader, model,mean,std)
        print("RFNE --- test1 error: %.5f %%, test2 error: %.5f %%" % (error1*100.0, error2*100.0),file = f)          

        error1, error2 = validate_PSNR(args, test1_loader, test2_loader, model,mean,std)
        print("PSNR --- test1 error: %.5f, test2 error: %.5f" % (error1, error2),file = f) 

        error1, error2 = validate_SSIM(args, test1_loader, test2_loader, model,mean,std)
        print("SSIM --- test1 error: %.5f, test2 error: %.5f" % (error1, error2),file = f) 

        if args.data_name == "nskt_16k" or args.data_name == "nskt_32k":
            phy_err1, phy_err2 = validate_phyLoss(args, test1_loader, test2_loader, model)
            print("Physics loss --- test1 error: %.8f, test2 error: %.8f" % (phy_err1, phy_err2),file=f) 

if __name__ =='__main__':
    main()