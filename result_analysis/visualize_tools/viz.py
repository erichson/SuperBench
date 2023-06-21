'''
Visualization function for bicubic downsampling:
    - This part may need users to manually modify the model load path. 
    - Users can customize their figures.
'''

import numpy as np
import torch
from torch import nn
import argparse
import matplotlib.pyplot as plt
import cmocean  
import matplotlib as mpl
from decimal import Decimal
import matplotlib.transforms as transforms
from src.data_loader_crop_visual import getData
from utils import *
from src.models import *
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def load_models_and_testloader(data_name, model_name, upscale_factor):
    '''load models and test datasets'''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_name == "cosmo":
        in_channels = 2
        out_channels = 2
        data_path = './datasets/cosmo_2048'
    elif data_name == "nskt_16k":
        in_channels = 3
        out_channels = 3
        data_path = './datasets/nskt16000_1024'
    elif data_name == "nskt_32k":
        in_channels = 3
        out_channels = 3
        data_path = './datasets/nskt32000_1024'
    elif data_name == "era5":
        in_channels = 3
        out_channels = 3
        data_path = './datasets/era5'

    resol, n_fields, n_train_samples, mean, std = get_data_info(data_name)
    window_size = 8
    height = (resol[0] // upscale_factor // window_size + 1) * window_size
    width = (resol[1] // upscale_factor // window_size + 1) * window_size
    model_list = {
            'subpixelCNN': subpixelCNN(in_channels, upscale_factor=upscale_factor, width=width, mean = mean,std = std),
            'SRCNN': SRCNN(in_channels, upscale_factor,mean,std),
            'EDSR': EDSR(in_channels, 64, 16, upscale_factor, mean, std),
            'WDSR': WDSR(in_channels, out_channels, 32, 18, upscale_factor, mean, std),
            'SwinIR': SwinIR(upscale=upscale_factor, in_chans=in_channels, img_size=(height, width),
                    window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
    }

    model = model_list[model_name]
    model = torch.nn.DataParallel(model)
    if model_name == 'bicubic':
        print('Using bicubic interpolation...')
    else: 
        lr = 0.001 if model_name == "SRCNN" or model_name == "subpixelCNN" else 0.0001

        if model_name == "SwinIR" or model_name == "SRCNN" or model_name == "subpixelCNN":
            #TODO: bicubic could be replace with uniform_downsample
            model_path = 'results/model_' + str(model_name) + '_' + str(data_name) + '_' + str(upscale_factor) + '_' + str(lr) + '_' + 'bicubic' +'_' + str(0.0) + '_' + str(5544) + '.pt'
        else: 
            model_path = 'results/model_' + str(model_name) + '_' + str(data_name) + '_' + str(upscale_factor) + '_' + str(lr) + '_' + 'bicubic' + '_' + str(5544) + '.pt'

        model = load_checkpoint(model, model_path)
        model = model.to(device)  

    # get data
    _, _, _, test1_loader, test2_loader = getData(
                                        data_name = data_name, 
                                        data_path = data_path, 
                                        upscale_factor = upscale_factor,
                                        noise_ratio = 0.0, 
                                        crop_size = 128, 
                                        method = 'bicubic', 
                                        batch_size = 1,
                                        n_patches =8, 
                                        std = std) # manually set batch_size = 1 for easy locate snapshot_num
    return model, test2_loader


def get_one_image(model, testloader, snapshot_num, channel_num):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for index, (data,target) in enumerate(testloader):
            data, target = data.to(device).float(), target.to(device).float()
            if index == snapshot_num:
                output = model(data)
                break
    data = data[:,channel_num,...].squeeze().detach().cpu().numpy()
    target = target[:,channel_num,...].squeeze().detach().cpu().numpy()
    output = output[:,channel_num,...].squeeze().detach().cpu().numpy()
    return data, target, output


def get_lim(data_name, model_name_list, upcale_factor, snapshot_num, channel_num):
    '''
    Get the min and max of all the snapshots. The goal is to make the plotted snapshots have the same colorbar range. 
    '''

    i = 2
    data_list = [] # 8 elements with each shape of [h,w]

    for name in model_name_list:
        print(name)
        model, data = load_models_and_testloader(data_name=data_name, model_name=name, upscale_factor=upcale_factor)
        
        if i == 2:
            # LR
            data_list.append(get_one_image(model, data, snapshot_num, channel_num)[0])

            # HR
            data_list.append(get_one_image(model, data, snapshot_num, channel_num)[1])

        data_list.append(get_one_image(model, data, snapshot_num, channel_num)[2])
        i += 1

    min_lim_list, max_lim_list = [], []
    for i in range(len(data_list)):
        min_lim_list.append(np.min(data_list[i]))
        max_lim_list.append(np.max(data_list[i]))

    min_lim = float(np.min(np.array(min_lim_list)))
    max_lim = float(np.max(np.array(max_lim_list)))    

    min_lim_round = float(Decimal(min_lim).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP"))
    max_lim_round = float(Decimal(max_lim).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP"))    

    lim = []
    if (min_lim_round - min_lim) > 0:
        lim.append(min_lim_round - 0.1)
    else:
        lim.append(min_lim_round)

    if (max_lim_round - max_lim) < 0:
        lim.append(max_lim_round+0.1)
    else:
        lim.append(max_lim_round)

    return lim, data_list # [min_lim, max_lim]


def plot_all_image(
    data_name = "nskt_16k",
    upcale_factor=8, 
    snapshot_num=300,
    channel_num = 1,
    zoom_loc_x = (300,380),
    zoom_loc_y = (80,140),
    figsize=(12,6),
    cmap=cmocean.cm.balance):

    model_name_list = ['bicubic', 'SRCNN', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR']
    title_list = ['LR','HR','Bicubic', 'SRCNN', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR']

    # get data and data range
    [vmin, vmax], data_list = get_lim(data_name, model_name_list, upcale_factor, snapshot_num, channel_num)
    vmean = (vmin + vmax) / 2.0
    vmean = float(Decimal(vmean).quantize(Decimal("0.1"), rounding = "ROUND_HALF_UP")) 
    print('The consistent min, mean and max are: ', vmin, vmean, vmax)

    # setup the figure definition
    fc = "none"
    if data_name == 'era5':
        font_size = 24 #16
        label_size = 20 # 14
        ec = "0.3" # "0.9" for cosmo, "0.3" for others
        box_color = 'k'
    elif data_name == 'cosmo':
        font_size = 14
        label_size = 14
        ec = "0.6"
        box_color = 'gainsboro'
    else:
        font_size = 14 
        label_size = 14 
        ec = "0.3"
        box_color = 'k'

    fig, axs = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=figsize,
        gridspec_kw={"width_ratios":[1,1,1,1,0.05]})

    for i in range(len(data_list)):
        if i == 0:
            # LR
            axs[0,0].imshow(data_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0,0].set_title(title_list[0], fontsize=font_size, weight='bold')
            axs[0,0].set_axis_off()
            # Draw zoom in 
            axins = zoomed_inset_axes(axs[0,0], 6, loc='lower left')    
            axins.imshow(data_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axins.set_xlim(tuple(val // upcale_factor for val in zoom_loc_x))
            axins.set_ylim(tuple(val // upcale_factor for val in zoom_loc_y))
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            _patch, pp1, pp2 = mark_inset(axs[0,0], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color) 
            pp1.loc1, pp1.loc2 = 2, 3  # inset corner 2 to origin corner 3 (would expect 2)
            pp2.loc1, pp2.loc2 = 4, 1  # inset corner 4 to origin corner 1 (would expect 4)
            plt.draw()
        else:
            im = axs[i//4,i%4].imshow(data_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[i//4,i%4].set_title(title_list[i], fontsize=font_size, weight = 'bold')
            axs[i//4,i%4].set_axis_off()
            # zoom in
            axins = zoomed_inset_axes(axs[i//4,i%4], 6, loc='lower left')    
            axins.imshow(data_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axins.set_xlim(zoom_loc_x)
            axins.set_ylim(zoom_loc_y)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            _patch, pp1, pp2 = mark_inset(axs[i//4,i%4], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color)
            pp1.loc1, pp1.loc2 = 2, 3  
            pp2.loc1, pp2.loc2 = 4, 1  
            plt.draw()

    # draw colorbar
    cbar = fig.colorbar(im, cax=axs[0,4], fraction=0.046, pad=0.04, extend='both')
    cbar.ax.tick_params(labelsize=label_size)
    cbar = fig.colorbar(im, cax=axs[1,4], fraction=0.046, pad=0.04, extend='both')
    cbar.ax.tick_params(labelsize=label_size)

    fig.tight_layout()
    
    fig.savefig(data_name+"_"+str(upcale_factor)+".png", dpi=300, bbox_inches='tight', transparent=False)
    fig.savefig(data_name+"_"+str(upcale_factor)+".pdf", bbox_inches='tight', transparent=False)

    return True


if __name__ == "__main__":
    '''
    data_name_list = ['nskt_16k','nskt_32k','cosmo','era5']
    '''

    # nskt 16k, spatial domain is [1024,1024]
    zoom_loc_x = (680, 760)
    zoom_loc_y = (260, 180)
    plot_all_image(data_name = "nskt_16k",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 2,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (10,5.5),
                    cmap = cmocean.cm.balance)

    plot_all_image(data_name = "nskt_16k",
                    upcale_factor = 16, 
                    snapshot_num = 55,
                    channel_num = 2,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (10,5.5),
                    cmap = cmocean.cm.balance)

    # nskt 32k, spatial domain is [1024,1024]
    zoom_loc_x = (860, 940)
    zoom_loc_y = (500, 420)
    plot_all_image(data_name = "nskt_32k",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 2,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (10,5.5),
                    cmap = cmocean.cm.balance)

    plot_all_image(data_name = "nskt_32k",
                    upcale_factor = 16, 
                    snapshot_num = 55,
                    channel_num = 2,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (10,5.5),
                    cmap = cmocean.cm.balance)

    # cosmology data, spatial domain is [2048,2048]
    zoom_loc_x = (1520, 1680)
    zoom_loc_y = (520, 360)
    plot_all_image(data_name = "cosmo",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (10,5.5),
                    cmap = cmocean.cm.thermal)

    plot_all_image(data_name = "cosmo",
                    upcale_factor = 16, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (10,5.5),
                    cmap = cmocean.cm.thermal)

    # weather data, spatial domain is [720,1440]
    zoom_loc_x = (820, 900) 
    zoom_loc_y = (230, 150)

    plot_all_image(data_name = "era5",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (18,5.5),
                    cmap = 'coolwarm')

    plot_all_image(data_name = "era5",
                    upcale_factor = 16, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (18,5.5),
                    cmap = 'coolwarm')

