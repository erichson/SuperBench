import numpy as np
import torch
from torch import nn
import argparse
from tqdm import tqdm

from src.get_data import getData
from src.models import *


parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--data', type=str, default='DoubleGyre', help='dataset')
parser.add_argument('--model_path', type=str, default='shallowDecoder', help='saved model')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

args = parser.parse_args()
print(args)

print(args.device)

#******************************************************************************
# Get data
#******************************************************************************
_, test1_loader, _, test2_loader, _ = getData(args.data, test_bs=args.batch_size)

# for inp, label in test2_loader:
#     print('{}:{}'.format(inp.shape, label.shape,))
#     break

#==============================================================================
# Get model
#==============================================================================
if args.data == 'isoflow4':
    input_size = [64, 64] 
    output_size = [256, 256]
elif args.data == 'isoflow8':
    input_size = [32, 32] 
    output_size = [256, 256]
    
elif args.data == 'doublegyre4':
    input_size = [112, 48] 
    output_size = [448, 192]
elif args.data == 'doublegyre8':
    input_size = [56, 24] 
    output_size = [448, 192]
    
elif args.data == 'rbc4':
    input_size = [128, 128] 
    output_size = [512, 512]        
elif args.data == 'rbc8':
    input_size = [64, 64] 
    output_size = [512, 512]    

elif args.data == 'rbcsc4':
    input_size = [128, 128] 
    output_size = [512, 512]        
elif args.data == 'rbcsc8':
    input_size = [64, 64] 
    output_size = [512, 512] 
    
elif args.data == 'sst4':
    input_size = [64, 128] 
    output_size = [256, 512]        
elif args.data == 'sst8':
    input_size = [32, 64] 
    output_size = [256, 512]    
    
elif args.data == 'sst4d':
    input_size = [64, 128] 
    output_size = [256, 512]        
elif args.data == 'sst8d':
    input_size = [32, 64] 
    output_size = [256, 512]        
    

model = torch.load(args.model_path).to(args.device)
#model = torch.nn.DataParallel(model)
model.eval()



#==============================================================================
# Model summary
#==============================================================================
# print(model)    
print('**** Setup ****')
print('Total params Generator: %.3fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')    


#******************************************************************************
# Validate
#******************************************************************************
criterion = nn.MSELoss()

def validate_mse(val1_loader, val2_loader, model):
    c = 0
    error1 = 0
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 

        target = target.data.cpu()
        output = output.data.cpu()

        error1 += criterion(output, target) * data.shape[0]
        c += data.shape[0]
    error1 /= c

    c = 0
    error2 = 0
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 

        target = target.data.cpu()
        output = output.data.cpu()
        
        error2 += criterion(output, target) * data.shape[0]
        c += data.shape[0]

    error2 /= c

    return error1.item(), error2.item()


def validate_RFNE(val1_loader, val2_loader, model):
    '''Relative Frobenius norm error (RFNE)'''
    error1 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 

        target = target.data.cpu().numpy().reshape(target.shape[0],-1)
        output = output.data.cpu().numpy().reshape(output.shape[0],-1)
    
        errors = [np.linalg.norm(target[i]-output[i])/np.linalg.norm(output[i]) for i in range(target.shape[0])]
        error1 += np.sum(errors)
        c += data.shape[0]
    error1 /= c

    error2 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 

        target = target.data.cpu().numpy().reshape(target.shape[0],-1)
        output = output.data.cpu().numpy().reshape(output.shape[0],-1)
    
        errors = [np.linalg.norm(target[i]-output[i])/np.linalg.norm(output[i]) for i in range(target.shape[0])]
        error2 += np.sum(errors)
        c += data.shape[0]
    error2 /= c
    
    return error1.item(), error2.item()



def validate_RINE(val1_loader, val2_loader, model):
    '''Relative infinity norm error (RINE)'''
    error1 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu().numpy().reshape(target.shape[0],-1)
        output = output.data.cpu().numpy().reshape(output.shape[0],-1)
        errors = [np.linalg.norm(target[i]-output[i], ord=np.inf)/np.linalg.norm(output[i], ord=np.inf) for i in range(target.shape[0])]
        error1 += np.sum(errors)
        c += data.shape[0]
    error1 /= c

    error2 = 0
    c = 0  
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu().numpy()
        output = output.data.cpu().numpy()
        
        target = target.reshape(target.shape[0],-1)
        output = output.reshape(output.shape[0],-1)
        errors = [np.linalg.norm(target[i]-output[i], ord=np.inf)/np.linalg.norm(output[i], ord=np.inf) for i in range(target.shape[0])]
        error2 += np.sum(errors)
        c += data.shape[0]
    error2 /= c
    
    return error1.item(), error2.item()

def validate_PSNR(val1_loader, val2_loader, model):
    '''Peak signal-to-noise ratio (PSNR)'''
    # install torchmetrics first: conda install -c conda-forge torchmetrics
    from torchmetrics import PeakSignalNoiseRatio
    psnr = PeakSignalNoiseRatio()
    
    error1 = 0
    c = 0      
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu()
        output = output.data.cpu()
        
        error1 = psnr(target, output)
    error1 = psnr.compute()

    psnr = PeakSignalNoiseRatio()
    error2 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu()
        output = output.data.cpu()    
        
        error2 += psnr(target, output)
    error2 = psnr.compute()

    return error1.item(), error2.item()

def validate_SSIM(val1_loader, val2_loader, model):
    '''Structual Similarity Index Measure (SSIM)'''
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure()
    
    error1 = 0
    c = 0      
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu()
        output = output.data.cpu()    
        
        error1 = ssim(target, output)
    error1 = ssim.compute()

    ssim = StructuralSimilarityIndexMeasure()
    error2 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu()
        output = output.data.cpu()
        
        error2 += ssim(target, output)
    error2 = ssim.compute()

    return error1.item(), error2.item()


def validate_multi_scale_SSIM(val1_loader, val2_loader, model):
    '''Multi-Scale Structual Similarity Index Measure (multi-scale SSIM)'''
    from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    
    error1 = 0
    c = 0      
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu()
        output = output.data.cpu()    
        
        error1 = ms_ssim(target, output)
    error1 = ms_ssim.compute()

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure()
    error2 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu()
        output = output.data.cpu()    
        
        error2 += ms_ssim(target, output)
    error2 = ms_ssim.compute()

    return error1.item(), error2.item()


def validate_viz(val1_loader, val2_loader, model):
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        break
      
    target = target.data.cpu().numpy()
    output = output.data.cpu().numpy()
    
    target = target[:,0,:,:]
    output = output[:,0,:,:]

    import matplotlib.pyplot as plt
    import cmocean        

    fig = plt.figure(figsize = (10, 10))
    a = fig.add_subplot(121)
    a.set_axis_off()

    a.imshow(target[3].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    a = fig.add_subplot(122)
    a.set_axis_off()
    a.imshow(output[3].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    plt.savefig('interplolation.pdf')


    plot_nice_viz(target, data, save_as='groundtruth')
    plot_nice_viz(target, data, save_as='reconstructed')


    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        break
    
    target = target.data.cpu().numpy()
    output = output.data.cpu().numpy()
    target = target[:,0,:,:]
    output = output[:,0,:,:]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (10, 10))
    a = fig.add_subplot(121)
    a.set_axis_off()
    a.imshow(target[3].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    a = fig.add_subplot(122)
    a.set_axis_off()
    a.imshow(output[3].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    plt.savefig('extrapolation.pdf')




# =============== validate ======================
error1, error2 = validate_mse(test1_loader, test2_loader, model)
print("MSE --- test1 error: %.8f, test2 error: %.8f" % (error1, error2))      
            
error1, error2  = validate_RFNE(test1_loader, test2_loader, model)
print("RFNE --- test1 error: %.5f, test2 error: %.5f" % (error1*100, error2*100))      

error1, error2 = validate_RINE(test1_loader, test2_loader, model)
print("RINE --- test1 error: %.5f, test2 error: %.5f" % (error1*100, error2*100))       

error1, error2 = validate_PSNR(test1_loader, test2_loader, model)
print("PSNR --- test1 error: %.5f, test2 error: %.5f" % (error1, error2)) 

error1, error2 = validate_SSIM(test1_loader, test2_loader, model)
print("SSIM --- test1 error: %.5f, test2 error: %.5f" % (error1, error2)) 

error1, error2 = validate_multi_scale_SSIM(test1_loader, test2_loader, model)
print("Multi-scale SSIM --- test1 error: %.5f, test2 error: %.5f" % (error1, error2)) 

validate_viz(test1_loader, test2_loader, model)
