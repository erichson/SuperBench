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

def validate_all_metrics(args, test1_loader, test2_loader, model, mean, std):
    from torchmetrics import StructuralSimilarityIndexMeasure

    ssim = StructuralSimilarityIndexMeasure().to(args.device)
    rine1, rine2, rfne1, rfne2, psnr1, psnr2, ssim1, ssim2,mse1,mse2,mae1,mae2 = [], [], [], [], [], [], [], [],[],[],[],[]

    # Helper function for PSNR
    def compute_psnr(true, pred):
        mse = torch.mean((true - pred) ** 2)
        if mse == 0:
            return float('inf')
        max_value = torch.max(true)
        return 20 * torch.log10(max_value / torch.sqrt(mse))

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

# def validate_RINE(args, test1_loader, test2_loader, model,mean,std):
#     '''Relative infinity norm error (RINE)'''
#     # calculate the RINE of each snapshot and then average
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate((test1_loader)):
#             data, target = data.to(args.device).float(), target.to(args.device).float() # [b,c,h,w]
#             output = model(data) 
#             output = normalize(args,output,mean,std)
#             target = normalize(args,target,mean,std)
#             # calculate infinity norm of each snapshot
#             err_ine = torch.norm((target-output), p=np.inf,dim = (-1,-2))/torch.norm(target,p=np.inf,dim = (-1,-2))

#     ine1 = err_ine.mean().item()
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate((test2_loader)):
#             data, target = data.to(args.device).float(), target.to(args.device).float()
#             output = model(data)  
#             # calculate infinity norm of each snapshot
#             output = normalize(args,output,mean,std)
#             target = normalize(args,target,mean,std)
#             err_ine = torch.norm(target-output, p=np.inf,dim = (-1,-2))/torch.norm(target,p=np.inf,dim = (-1,-2))
#     ine2 = err_ine.mean().item()
#     return ine1, ine2 


# def validate_RFNE(args, test1_loader, test2_loader, model,mean,std):
#     '''Relative Frobenius norm error (RFNE)'''
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate((test1_loader)):
#             data, target = data.to(args.device).float(), target.to(args.device).float()
#             output = model(data)   
#             output = normalize(args,output,mean,std)
#             target = normalize(args,target,mean,std)
#             # calculate frobenius norm of each snapshot of each channel
#             err_rfne = torch.norm((target-output),p =2,dim=(-1,-2)) / torch.norm(target,p =2,dim=(-1,-2))
#     rfne1 =err_rfne.mean().item()

#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate((test2_loader)):
#             data, target = data.to(args.device).float(), target.to(args.device).float()
#             output = model(data) 
#             output = normalize(args,output,mean,std)
#             target = normalize(args,target,mean,std)
#             # calculate frobenius norm of each snapshot
#             err_rfne = torch.norm((target-output),p =2,dim=(-1,-2)) / torch.norm(target,p =2,dim=(-1,-2))

#     #fne2 = np.array(fne2).mean()
#     rfne2 = err_rfne.mean().item()

#     return rfne1, rfne2



# def psnr(true, pred):
#     mse = torch.mean((true - pred) ** 2)
#     if mse == 0:
#         return float('inf')
#     max_value = torch.max(true)
#     return 20 * torch.log10(max_value / torch.sqrt(mse))


# def validate_PSNR(args, test1_loader, test2_loader, model,mean,std):
#     '''Peak signal-to-noise ratio (PSNR)'''
    
#     error1 = []   
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate((test1_loader)):
#             data, target = data.to(args.device).float(), target.to(args.device).float()
#             output = model(data) 
#             output = normalize(args,output,mean,std)
#             target = normalize(args,target,mean,std)
#             # calculate PSNR of each snapshot and then average (Change to channel-wise)
#             for i in range(target.shape[0]):
#                 for j in range(target.shape[1]):
#                     err_psnr = psnr(target[i,j,...], output[i,j,...])
#                     error1.append(err_psnr)
#     error1 = torch.mean(torch.stack(error1)).item()

#     error2 = []  
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate((test2_loader)):
#             data, target = data.to(args.device).float(), target.to(args.device).float()
#             output = model(data) 
#             output = normalize(args,output,mean,std)
#             target = normalize(args,target,mean,std)
#             # calculate PSNR of each snapshot and then average
#             for i in range(target.shape[0]):
#                 for j in range(target.shape[1]):
#                     err_psnr = psnr(target[i,j,...], output[i,j,...])
#                     error2.append(err_psnr)
#     error2 = torch.mean(torch.stack(error2)).item()

#     return error1, error2


# def validate_SSIM(args, test1_loader, test2_loader, model,mean,std):
#         '''Structual Similarity Index Measure (SSIM)'''
#         from torchmetrics import StructuralSimilarityIndexMeasure
#         ssim = StructuralSimilarityIndexMeasure().to(args.device)
        
#         error1 = []
#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate((test1_loader)):
#                 data, target = data.to(args.device).float(), target.to(args.device).float()
#                 output = model(data) 
                
#                 output = normalize(args,output,mean,std)
#                 target = normalize(args,target,mean,std)
#                 for i in range(target.shape[0]):
#                     for j in range(target.shape[1]):
#                         err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
#                         error1.append(err_ssim.cpu())

#         # averaged SSIM
#         err1 = torch.mean(torch.stack(error1)).item()

#         error2 = []
#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate((test2_loader)):
#                 data, target = data.to(args.device).float(), target.to(args.device).float()
#                 output = model(data) 
#                 output = normalize(args,output,mean,std)
#                 target = normalize(args,target,mean,std)                
#                 for i in range(target.shape[0]):
#                     for j in range(target.shape[1]):
#                         err_ssim = ssim(target[i:(i+1),j:(j+1),...], output[i:(i+1),j:(j+1),...])
#                         error2.append(err_ssim.cpu())

#         err2 = torch.mean(torch.stack(error2)).item()

#         return err1, err2

    
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
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
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
            'EDSR': EDSR(args.in_channels, 64, 16, args.upscale_factor, mean, std),
            'WDSR': WDSR(args.in_channels,args.in_channels, 32,18, args.upscale_factor, mean, std),
            'SwinIR': SwinIR(upscale=args.upscale_factor, in_chans=args.in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
            'Bicubic': Bicubic(args.upscale_factor),
    }

    model = model_list[args.model]
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
    all_results[key]["metrics"]["Infinity"] = {'test1 error': INE[0], 'test2 error': INE[1]}

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
    if args.data_name in ["nskt_16k", "nskt_32k","nskt_16k_sim","nskt_32k_sim"]:
        phy_err1, phy_err2 = validate_phyLoss(args, test1_loader, test2_loader, model)
        all_results[key]["metrics"]["Physics"] = {'test1 error': phy_err1, 'test2 error': phy_err2}

    # Serialize the updated results list to the JSON file
    with open("normed_eval.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # =============== validate ======================
    # with open("result.txt", "a") as f:
    #     print(" model" + str(args.model) + " data: " + str(args.data_name)+ "  method: " + str(args.method) +" scale factor " + str(args.upscale_factor) + " noise ratio: " + str(args.noise_ratio),file = f)
    #     ine1, ine2 = validate_RINE(args, test1_loader, test2_loader, model,mean,std)
    #     print("Infinity norm --- test1 error: %.8f, test2 error: %.8f" % (ine1, ine2),file = f) 

    #     error1, error2 = validate_RFNE(args, test1_loader, test2_loader, model,mean,std)
    #     print("RFNE --- test1 error: %.5f %%, test2 error: %.5f %%" % (error1*100.0, error2*100.0),file = f)          

    #     error1, error2 = validate_PSNR(args, test1_loader, test2_loader, model,mean,std)
    #     print("PSNR --- test1 error: %.5f, test2 error: %.5f" % (error1, error2),file = f) 

    #     error1, error2 = validate_SSIM(args, test1_loader, test2_loader, model,mean,std)
    #     print("SSIM --- test1 error: %.5f, test2 error: %.5f" % (error1, error2),file = f) 

    #     if args.data_name == "nskt_16k" or args.data_name == "nskt_32k":
    #         phy_err1, phy_err2 = validate_phyLoss(args, test1_loader, test2_loader, model)
    #         print("Physics loss --- test1 error: %.8f, test2 error: %.8f" % (phy_err1, phy_err2),file=f) 

if __name__ =='__main__':
    main()