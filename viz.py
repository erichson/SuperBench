import numpy as np
import torch
from torch import nn
import argparse
from tqdm import tqdm

from src.get_data import getData
from src.models import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import cmocean     
from matplotlib import cm  
from matplotlib.colors import ListedColormap,LinearSegmentedColormap 

parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--data', type=str, default='isoflow4', help='dataset')
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


def plot_nice_viz(img, data, label):
    
    
    if data == 'sst8' or data == 'sst4' or data == 'sst8d' or data == 'sst4d':
        fig, a = plt.subplots(figsize=(10,7))    
        extent = (0, 0, 0, 0)
        a.set_xlim(120, 490)
        a.set_axis_off()
        a.imshow(img.reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
        axins = zoomed_inset_axes(a, 2, loc='lower left')    
        axins.imshow(img.reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
        axins.set_xlim(300, 380)
        axins.set_ylim(80, 130)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(a, axins, loc1=2, loc2=4, fc="none", ec="0.9", lw=1.5, color='k')
        plt.draw()
        #plt.tight_layout()
        plt.savefig(data + '_' + label + '.pdf')    
    
    if data == 'isoflow8' or data == 'isoflow4':
        
        # define top and bottom colormaps 
        top = cm.get_cmap('Oranges_r', 128) # r means reversed version
        bottom = cm.get_cmap('Blues', 128)# combine it all
        newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                               bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
        orange_blue = ListedColormap(newcolors, name='OrangeBlue')        
        
        
        fig, a = plt.subplots(figsize=(10,7))    
        extent = (0, 0, 0, 0)
        a.set_axis_off()
        a.imshow(img.reshape(output_size[0],output_size[1]), cmap=orange_blue)
        axins = zoomed_inset_axes(a, 2, loc='lower left')    
        axins.imshow(img.reshape(output_size[0],output_size[1]), cmap=orange_blue)
        axins.set_xlim(140, 190)
        axins.set_ylim(80, 130)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(a, axins, loc1=2, loc2=4, fc="none", ec="0.9", lw=1.5, color='k')
        plt.draw()
        #plt.tight_layout()
        plt.savefig(data + '_' + label + '.pdf')    
       
    
def validate_viz(val1_loader, val2_loader, model, data_name):
    for batch_idx, (data, target) in enumerate(val1_loader):
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
    plt.savefig('interplolation.pdf')


    plot_nice_viz(target[0], data_name, label='groundtruth_inter')
    plot_nice_viz(output[0], data_name, label='reconstructed_inter')


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
    a.imshow(target[3].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    a = fig.add_subplot(122)
    a.set_axis_off()
    a.imshow(output[3].reshape(output_size[0],output_size[1]), cmap=cmocean.cm.balance)
    plt.savefig('extrapolation.pdf')

    plot_nice_viz(target[0], data_name, label='groundtruth_extra')
    plot_nice_viz(output[0], data_name, label='reconstructed_extra')


# =============== validate ======================
validate_viz(test1_loader, test2_loader, model, args.data)
