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
parser.add_argument('--batch_size', type=int, default=300, help='batch size')


args = parser.parse_args()
print(args)



#******************************************************************************
# Get data
#******************************************************************************
train_loader, test1_loader, val1_loader, test2_loader, val2_loader = getData(args.data, test_bs=args.batch_size)

for inp, label in test2_loader:
    print('{}:{}'.format(inp.shape, label.shape,))
    break

#==============================================================================
# Get model
#==============================================================================
if args.data == 'isoflow':
    input_size = [64, 64] 
    output_size = [256, 256]
elif args.data == 'DoubleGyre':
    input_size = [112, 48] 
    output_size = [448, 192]
elif args.data == 'RBC':
    input_size = [32, 32] 
    output_size = [256, 256]        
    

model = torch.load(args.model_path).to(args.device)

#==============================================================================
# Model summary
#==============================================================================
# print(model)    
# print('**** Setup ****')
# print('Total params Generator: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
# print('************')    


#******************************************************************************
# Validate
#******************************************************************************
criterion = nn.MSELoss(reduction='sum').to(args.device)

def validate_mse(val1_loader, val2_loader, model):
    c = 0
    error1 = 0
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        error1 += criterion(output, target)
        c += data.shape[0]
    error1 /= c

    c = 0
    error2 = 0
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        error2 += criterion(output, target)
        c += data.shape[0]
    error2 /= c

    return error1.item(), error2.item()


def validate_MSPE(val1_loader, val2_loader, model):

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



def validate_MAPE(val1_loader, val2_loader, model):
    from sklearn.metrics import mean_absolute_percentage_error
    
    error1 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
        target = target.data.cpu().numpy().reshape(target.shape[0],-1)
        output = output.data.cpu().numpy().reshape(output.shape[0],-1)
        error1 += mean_absolute_percentage_error(target, output) * data.shape[0]
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
        error2 += mean_absolute_percentage_error(target, output) * data.shape[0]
        c += data.shape[0]
    error2 /= c
    
    return error1.item(), error2.item()

def validate_PSNR(val1_loader, val2_loader, model):
    # load torchmetrics first: conda install -c conda-forge torchmetrics
    from torchmetrics import PeakSignalNoiseRatio
    psnr = PeakSignalNoiseRatio().to(args.device)
    
    error1 = 0
    c = 0      
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
    
        error1 += psnr(target, output) * data.shape[0]
        c += data.shape[0]
    error1 /= c

    error2 = 0
    c = 0    
    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
    
        error2 += psnr(target, output)
        c += data.shape[0]
    error2 /= c

    return error1.item(), error2.item()



def validate_viz(val1_loader, val2_loader, model):
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
      
    target = target.data.cpu().numpy()
    output = output.data.cpu().numpy()
    
    target = target[:,0,:,:]
    output = output[:,0,:,:]

    import matplotlib.pyplot as plt
    import cmocean        

    fig = plt.figure(figsize = (10, 10))
    a = fig.add_subplot(121)
    a.set_axis_off()
    vmin=target[0].min()
    vmax=target[0].max()
    a.imshow(target[0].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    a = fig.add_subplot(122)
    a.set_axis_off()
    a.imshow(output[0].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    plt.savefig('interplolation.pdf')


    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        
    target = target.data.cpu().numpy()
    output = output.data.cpu().numpy()
    target = target[:,0,:,:]
    output = output[:,0,:,:]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (10, 10))
    a = fig.add_subplot(121)
    a.set_axis_off()
    vmin=target[20].min()
    vmax=target[20].max()
    a.imshow(target[20].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    a = fig.add_subplot(122)
    a.set_axis_off()
    a.imshow(output[20].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    plt.savefig('extrapolation.pdf')


# =============== validate ======================
error1, error2 = validate_mse(test1_loader, test2_loader, model)
print("MSE --- test1 error: %.8f, test2 error: %.8f" % (error1, error2))      
            
error1, error2  = validate_MSPE(test1_loader, test2_loader, model)
print("MSPE --- test1 error: %.5f, test2 error: %.5f" % (error1*100, error2*100))      

error1, error2 = validate_MAPE(test1_loader, test2_loader, model)
print("MAPE --- test1 error: %.5f, test2 error: %.5f" % (error1*100, error2*100))       

error1, error2 = validate_PSNR(test1_loader, test2_loader, model)
print("PSNR --- test1 error: %.5f, test2 error: %.5f" % (error1, error2)) 

validate_viz(test1_loader, test2_loader, model)
    


