import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
import cmocean


def plot_all_image42(
    data_name = "nskt_16k",
    upcale_factor=8, 
    snapshot_num=16,
    channel_num = -1,
    zoom_loc_x = (300,380),
    zoom_loc_y = (80,140),
    figsize=(8,8),
    cmap=cmocean.cm.balance):
    if data_name !='era5':
        model_saved_list = ['Bicubic', 'SRCNN','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['LR','HR','Bicubic', 'SRCNN','subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]
    else:
        model_saved_list = ['Bicubic', 'SRCNN','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['LR','HR','Bicubic', 'SRCNN', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]


    # get data and data range
    path = "/pscratch/sd/j/junyi012/superbench_v2/plot_buffer/"
    lr = np.load(path+data_name+"_"+str(upcale_factor)+"_lr.npy")
    hr = np.load(path+data_name+"_"+str(upcale_factor)+"_hr.npy")
    lr = lr[snapshot_num,channel_num]
    hr = hr[snapshot_num,channel_num]
    
    vmin = np.min(hr)
    vmax = np.max(hr)
    pred_list = [lr,hr]+[np.load(path+f"{data_name}_{model_name}_{upcale_factor}_pred_b{snapshot_num}c{channel_num}.npy")[0,channel_num] for model_name in model_saved_list]

    print(len(pred_list))
    print(lr.shape, hr.shape,pred_list[0].shape,pred_list[1].shape,pred_list[2].shape)
    # fig = plt.figure(figsize=(9.75, 3))

    # grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
    #                 nrows_ncols=(3,3),
    #                 axes_pad=0.15,
    #                 share_all=True,
    #                 cbar_location="right",
    #                 cbar_mode="single",
    #                 cbar_size="7%",
    #                 cbar_pad=0.15,
    #                 )

    # # Add data to image grid
    # for ax,img in zip(grid, pred_list):
    #     im = ax.imshow(img, vmin=vmin, vmax=vmax)
    #     ax.set_axis_off()
    # # Colorbar
    # ax.cax.colorbar(im)
    # ax.cax.toggle_label(True)
    # fig.savefig("Baseline_visualization_" + data_name+"_"+str(upcale_factor)+".png", dpi=300, bbox_inches='tight', transparent=False)
    # setup the figure definition
    fc = "none"
    if data_name == 'era5':
        font_size = 10 #16
        label_size =  9 # 14
        ec = "0.3" # "0.9" for cosmo, "0.3" for others
        box_color = 'k'
        zoom_in_factor = 4.5
        pred_list2 = []
        for data in pred_list:
            if data.shape[0]>=data.shape[1]:
                dim = data.shape[1]
            else:
                dim = data.shape[0]
            pred_list2.append(data[:,-1-dim:-1])

        pred_list = pred_list2
    elif data_name == 'cosmo' or data_name =="cosmo_sim_8":
        font_size = 10
        label_size = 9
        ec = "0.6"
        box_color = 'gainsboro'
        zoom_in_factor = 11
    else:
        font_size = 10
        label_size = 9
        ec = "0.3"
        box_color = 'k'
        zoom_in_factor = 11

    fig, axs = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=figsize,
        constrained_layout=True,
        # gridspec_kw={"width_ratios":[1,1,1,0.05]}
        )

    for i in range(len(pred_list)):
        if i == 0:
            # LR
            axs[0,0].imshow(pred_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0,0].set_title(title_list[0], fontsize=font_size)
            axs[0,0].set_axis_off()
            # Draw zoom in 
            axins = zoomed_inset_axes(axs[0,0], zoom_in_factor, loc='lower left')    
            axins.imshow(pred_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axins.set_xlim(tuple(val // upcale_factor for val in zoom_loc_x))
            axins.set_ylim(tuple(val // upcale_factor for val in zoom_loc_y))
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            _patch, pp1, pp2 = mark_inset(axs[0,0], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color) 
            pp1.loc1, pp1.loc2 = 2, 3  # inset corner 2 to origin corner 3 (would expect 2)
            pp2.loc1, pp2.loc2 = 4, 1  # inset corner 4 to origin corner 1 (would expect 4)
            plt.draw()
        else:
            im = axs[i//4,i%4].imshow(pred_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[i//4,i%4].set_title(title_list[i], fontsize=font_size)
            axs[i//4,i%4].set_axis_off()
            # zoom in
            axins = zoomed_inset_axes(axs[i//4,i%4], zoom_in_factor, loc='lower left')    
            axins.imshow(pred_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axins.set_xlim(zoom_loc_x)
            axins.set_ylim(zoom_loc_y)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            _patch, pp1, pp2 = mark_inset(axs[i//4,i%4], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color)
            pp1.loc1, pp1.loc2 = 2, 3  
            pp2.loc1, pp2.loc2 = 4, 1  
            plt.draw()

    # draw colorbar
    # cbar = fig.colorbar(im, cax=axs[0,3], fraction=0.046, pad=0.04, extend='both')
    # cbar.ax.tick_params(labelsize=label_size)
    # cbar = fig.colorbar(im, cax=axs[1,3], fraction=0.046, pad=0.04, extend='both')
    # cbar.ax.tick_params(labelsize=label_size)
    # cbar = fig.colorbar(im, cax=axs[2,3], fraction=0.046, pad=0.04, extend='both')
    # cbar.ax.tick_params(labelsize=label_size)
    # fig.tight_layout()
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    normalizer = Normalize(vmin, vmax)
    im = cm.ScalarMappable(norm=normalizer,cmap=cmap)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(),fraction=0.04666, pad=0.02, extend='both',aspect=30) # larger aspect value make cbar thinner
    cbar.ax.tick_params(labelsize=label_size)
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    # fig.tight_layout(w_pad=0.25,h_pad=0.25,pad=0.25)
    # fig.tight_layout()
    fig.savefig(data_name+"_"+str(upcale_factor)+"_24"+".png", dpi=300, bbox_inches='tight', transparent=False)
    fig.savefig(data_name+"_"+str(upcale_factor)+"_24"+".pdf", bbox_inches='tight', transparent=False)

    return True

def plot_all_image(
    data_name = "nskt_16k",
    upcale_factor=8, 
    snapshot_num=16,
    channel_num = -1,
    zoom_loc_x = (300,380),
    zoom_loc_y = (80,140),
    figsize=(8,8),
    cmap=cmocean.cm.balance):
    if data_name !='era5':
        model_saved_list = ['Bicubic', 'SRCNN', 'FNO2D_patch','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['LR','HR','Bicubic', 'SRCNN','FNO$^*$', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]
    else:
        model_saved_list = ['Bicubic', 'SRCNN', 'FNO2D','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['LR','HR','Bicubic', 'SRCNN','FNO', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]


    # get data and data range
    path = "/pscratch/sd/j/junyi012/superbench_v2/plot_buffer/"
    lr = np.load(path+data_name+"_"+str(upcale_factor)+"_lr.npy")
    hr = np.load(path+data_name+"_"+str(upcale_factor)+"_hr.npy")
    lr = lr[snapshot_num,channel_num]
    hr = hr[snapshot_num,channel_num]
    
    vmin = np.min(hr)
    vmax = np.max(hr)
    pred_list = [lr,hr]+[np.load(path+f"{data_name}_{model_name}_{upcale_factor}_pred_b{snapshot_num}c{channel_num}.npy")[0,channel_num] for model_name in model_saved_list]

    print(len(pred_list))
    print(lr.shape, hr.shape,pred_list[0].shape,pred_list[1].shape,pred_list[2].shape)
    # fig = plt.figure(figsize=(9.75, 3))

    # grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
    #                 nrows_ncols=(3,3),
    #                 axes_pad=0.15,
    #                 share_all=True,
    #                 cbar_location="right",
    #                 cbar_mode="single",
    #                 cbar_size="7%",
    #                 cbar_pad=0.15,
    #                 )

    # # Add data to image grid
    # for ax,img in zip(grid, pred_list):
    #     im = ax.imshow(img, vmin=vmin, vmax=vmax)
    #     ax.set_axis_off()
    # # Colorbar
    # ax.cax.colorbar(im)
    # ax.cax.toggle_label(True)
    # fig.savefig("Baseline_visualization_" + data_name+"_"+str(upcale_factor)+".png", dpi=300, bbox_inches='tight', transparent=False)
    # setup the figure definition
    fc = "none"
    if data_name == 'era5':
        font_size = 10 #16
        label_size =  9 # 14
        ec = "0.3" # "0.9" for cosmo, "0.3" for others
        box_color = 'k'
        zoom_in_factor = 4.5
        pred_list2 = []
        for data in pred_list:
            if data.shape[0]>=data.shape[1]:
                dim = data.shape[1]
            else:
                dim = data.shape[0]
            pred_list2.append(data[:,-1-dim:-1])

        pred_list = pred_list2
    elif data_name == 'cosmo' or data_name =="cosmo_sim_8":
        font_size = 10
        label_size = 9
        ec = "0.6"
        box_color = 'gainsboro'
        zoom_in_factor = 11
    else:
        font_size = 10
        label_size = 9
        ec = "0.3"
        box_color = 'k'
        zoom_in_factor = 11

    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=figsize,
        constrained_layout=True,
        # gridspec_kw={"width_ratios":[1,1,1,0.05]}
        )

    for i in range(len(pred_list)):
        if i == 0:
            # LR
            axs[0,0].imshow(pred_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0,0].set_title(title_list[0], fontsize=font_size)
            axs[0,0].set_axis_off()
            # Draw zoom in 
            axins = zoomed_inset_axes(axs[0,0], zoom_in_factor, loc='lower left')    
            axins.imshow(pred_list[0], vmin=vmin, vmax=vmax, cmap=cmap)
            axins.set_xlim(tuple(val // upcale_factor for val in zoom_loc_x))
            axins.set_ylim(tuple(val // upcale_factor for val in zoom_loc_y))
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            _patch, pp1, pp2 = mark_inset(axs[0,0], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color) 
            pp1.loc1, pp1.loc2 = 2, 3  # inset corner 2 to origin corner 3 (would expect 2)
            pp2.loc1, pp2.loc2 = 4, 1  # inset corner 4 to origin corner 1 (would expect 4)
            plt.draw()
        else:
            im = axs[i//3,i%3].imshow(pred_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axs[i//3,i%3].set_title(title_list[i], fontsize=font_size)
            axs[i//3,i%3].set_axis_off()
            # zoom in
            axins = zoomed_inset_axes(axs[i//3,i%3], zoom_in_factor, loc='lower left')    
            axins.imshow(pred_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
            axins.set_xlim(zoom_loc_x)
            axins.set_ylim(zoom_loc_y)
            plt.xticks(visible=False)
            plt.yticks(visible=False)
            _patch, pp1, pp2 = mark_inset(axs[i//3,i%3], axins, loc1=2, loc2=4, fc=fc, ec=ec, lw=1.0, color=box_color)
            pp1.loc1, pp1.loc2 = 2, 3  
            pp2.loc1, pp2.loc2 = 4, 1  
            plt.draw()

    # draw colorbar
    # cbar = fig.colorbar(im, cax=axs[0,3], fraction=0.046, pad=0.04, extend='both')
    # cbar.ax.tick_params(labelsize=label_size)
    # cbar = fig.colorbar(im, cax=axs[1,3], fraction=0.046, pad=0.04, extend='both')
    # cbar.ax.tick_params(labelsize=label_size)
    # cbar = fig.colorbar(im, cax=axs[2,3], fraction=0.046, pad=0.04, extend='both')
    # cbar.ax.tick_params(labelsize=label_size)
    # fig.tight_layout()
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    normalizer = Normalize(vmin, vmax)
    im = cm.ScalarMappable(norm=normalizer,cmap=cmap)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(),fraction=0.04666, pad=0.02, extend='both',aspect=30) # larger aspect value make cbar thinner
    cbar.ax.tick_params(labelsize=label_size)
    # fig.tight_layout(w_pad=0.25,h_pad=0.25,pad=0.25)
    # fig.tight_layout()
    fig.savefig(data_name+"_"+str(upcale_factor)+"_33"+".png", dpi=300, bbox_inches='tight', transparent=False)
    fig.savefig(data_name+"_"+str(upcale_factor)+"_33"+".pdf", bbox_inches='tight', transparent=False)

    return True

if __name__ == "__main__":
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
    # # nskt 16k, spatial domain is [1024,1024]
    zoom_loc_x = (1600, 1680)
    zoom_loc_y = (490, 410)
    import seaborn
    import cmocean
    plot_all_image(data_name = "nskt_16k",
                    upcale_factor = 8, 
                    snapshot_num = 16,
                    channel_num = -1,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.balance)
    plot_all_image(data_name = "nskt_16k",
                    upcale_factor = 16, 
                    snapshot_num = 16,
                    channel_num = -1,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.balance)
    

    # # nskt 32k, spatial domain is [1024,1024]
    zoom_loc_x = (1220, 1300)
    zoom_loc_y = (440, 360)
    plot_all_image(data_name = "nskt_32k",
                    upcale_factor = 8, 
                    snapshot_num = 16,
                    channel_num = -1,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.balance)

    plot_all_image(data_name = "nskt_32k",
                    upcale_factor = 16, 
                    snapshot_num = 16,
                    channel_num = -1,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.balance)

    # cosmology data, spatial domain is [2048,2048]
    zoom_loc_x = (620, 700)
    zoom_loc_y = (420, 340)
    plot_all_image(data_name = "cosmo",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.thermal)

    plot_all_image(data_name = "cosmo",
                    upcale_factor = 16, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.thermal)

    # # weather data, spatial domain is [720,1440]
    zoom_loc_x = (620, 700) 
    zoom_loc_y = (180, 100)

    plot_all_image(data_name = "era5",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = 'coolwarm')
    # plot_all_image42(data_name = "era5",
    #                 upcale_factor = 8, 
    #                 snapshot_num = 55,
    #                 channel_num = 0,
    #                 zoom_loc_x = zoom_loc_x,
    #                 zoom_loc_y = zoom_loc_y, 
    #                 figsize = (7.2,7/4*2),
    #                 cmap = 'coolwarm')
    # plot_all_image42(data_name = "era5",
    #                 upcale_factor = 16, 
    #                 snapshot_num = 55,
    #                 channel_num = 0,
    #                 zoom_loc_x = zoom_loc_x,
    #                 zoom_loc_y = zoom_loc_y, 
    #                 figsize = (7.1,7/4*2),
    #                 cmap = 'coolwarm')
                   
    plot_all_image(data_name = "era5",
                    upcale_factor = 16, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x,
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = 'coolwarm')
    zoom_loc_x = (1600, 1680)
    zoom_loc_y = (490, 410)
    import seaborn
    import cmocean
    plot_all_image(data_name = "nskt_16k_sim_4_v8",
                    upcale_factor = 4, 
                    snapshot_num = 12,
                    channel_num = -1,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.balance)
    zoom_loc_x = (1600, 1680)
    zoom_loc_y = (490, 410)
    plot_all_image(data_name = "nskt_32k_sim_4_v8",
                    upcale_factor = 4, 
                    snapshot_num = 12,
                    channel_num = -1,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.balance)
    zoom_loc_x = (1500, 1580)
    zoom_loc_y = (410, 330)
    plot_all_image(data_name = "cosmo_sim_8",
                    upcale_factor = 8, 
                    snapshot_num = 55,
                    channel_num = 0,
                    zoom_loc_x = zoom_loc_x, 
                    zoom_loc_y = zoom_loc_y, 
                    figsize = (7.2,7),
                    cmap = cmocean.cm.thermal)