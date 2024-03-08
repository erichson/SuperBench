import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
import pyfftw
import seaborn
import os
import cmocean


def plot_energy_spectrum_v2(
    data_name = "nskt_32k",
    upcale_factor=16, 
    snapshot_num=16, # this is a useless parameter (was used for debugging)
    zoom_in_factor=1,
    power=0,
    ):
    
    if data_name =='era5':
        model_saved_list = ['Bicubic', 'SRCNN', 'FNO2D','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['HR','Bicubic', 'SRCNN','FNO', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]
    else:
        model_saved_list = ['FNO2D_patch','WDSR', 'EDSR','SwinIR',"SwinIR_p_001","HR",]
        title_list = ['FNO$^*$','WDSR', 'EDSR','SwinIR',"SwinIR (Phy)", 'HR'] # color from light to dark ()
        # NOTE : HR has to be the last one; as the if condition in the for loop if i ==len(model_saved_list)-1:
    cmap= seaborn.color_palette('YlGnBu', n_colors=len(model_saved_list))
    cmap[-1]= 'r'
    # get data and data range
    path = "/pscratch/sd/j/junyi012/superbench_v2/eval_buffer/"

    fig,ax = plt.subplots(figsize=(3.4,3),constrained_layout=True)
    fontsize = 9
    axins = zoomed_inset_axes(ax,zoom_in_factor,loc='lower left') # [x0, y0, width, height]
    for i,model_name in enumerate(model_saved_list):
        if i ==len(model_saved_list)-1:
            if os.path.exists(path+f"energy_spectrum_v2_en_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy") == False:
                hr = np.load(path+f"{data_name}_{upcale_factor}_hr.npy")
                en, n = energy_spectrum_v2(hr[:,-2],hr[:,-1])
                np.save(path+f"energy_spectrum_v2_en_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy",en)
                k = np.linspace(1,n,n)
                np.save(path+f"energy_spectrum_v2_k_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy",k)
            else:
                en = np.load(path+f"energy_spectrum_v2_en_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy")
                k = np.load(path+f"energy_spectrum_v2_k_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy")
        else:
            if os.path.exists(path+f"energy_spectrum_v2_k_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy") == False:
                pred = np.load(path+f"{data_name}_{upcale_factor}_{model_name}_pred.npy")
                en, n = energy_spectrum_v2(pred[:,-2],pred[:,-1])
                np.save(path+f"energy_spectrum_v2_en_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy",en)
                k = np.linspace(1,n,n)
                np.save(path+f"energy_spectrum_v2_k_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy",k)
            else:
                en = np.load(path+f"energy_spectrum_v2_en_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy")
                k = np.load(path+f"energy_spectrum_v2_k_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy")
        ax.loglog(k,en[1:len(k)+1]*k**power,label=title_list[i],color=cmap[i])
        axins.loglog(k,en[1:len(k)+1]*k**power,label=title_list[i],color=cmap[i])
    ax.legend(fontsize=fontsize-2,bbox_to_anchor=(0.5, -0.45), loc='lower center',ncol=3)
    ax.set_xlabel("Wavenumber k",fontsize=fontsize)
    ax.set_ylabel("Engergy Spectrum E(k)",fontsize=fontsize)
    # plt.xticks(fontsize=fontsize-1)
    # plt.yticks(fontsize=fontsize-1)
    ax.set_xlim(1,200)
    # plt.ylim(1e-12,1)
    
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_major_locator(LogLocator(numticks=5))
    ax.xaxis.set_major_locator(LogLocator(numticks=3))
    _patch, pp1, pp2 = mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.3', lw=1.0, color='k') 
    axins.xaxis.set_tick_params(labelbottom=False)
    axins.yaxis.set_tick_params(labelleft=False)
    if power ==0:
        axins.set_xlim(40,100)
        axins.set_ylim(200,1000)
    else:
        axins.set_xlim(40,100)
        axins.set_ylim(200,1000)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.minorticks_off()
    fig.savefig(f"energy_spectrum_v2_{data_name}_{upcale_factor}_snpt{snapshot_num}_{power}.png",dpi=300,bbox_inches='tight')
    fig.savefig(f"energy_spectrum_v2_{data_name}_{upcale_factor}_snpt{snapshot_num}.pdf",bbox_inches='tight')
    return True

def plot_energy_spectrum(
    data_name = "nskt_32k",
    upcale_factor=16, 
    snapshot_num=16,
    cmap=cmocean.cm.balance):
    if data_name =='era5':
        model_saved_list = ['Bicubic', 'SRCNN', 'FNO2D','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['HR','Bicubic', 'SRCNN','FNO', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]
    else:
        model_saved_list = ["HR",'FNO2D_patch', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['HR','FNO$^*$', 'EDSR', 'WDSR', 'SwinIR',]


    # get data and data range
    path = "/pscratch/sd/j/junyi012/superbench_v2/eval_buffer/"

    fig = plt.figure(figsize=(4,3),constrained_layout=True)
    fontsize = 9
    for i,model_name in enumerate(model_saved_list):
        if i ==0:
            if os.path.exists(path+f"energy_spectrum_en_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy") == False:
                hr = np.load(path+f"{data_name}_{upcale_factor}_hr.npy")[snapshot_num,-1]
                en, n = energy_spectrum(2048,2048,hr)
                np.save(path+f"energy_spectrum_en_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy",en)
                k = np.linspace(1,n,n)
                np.save(path+f"energy_spectrum_k_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy",k)
            else:
                en = np.load(path+f"energy_spectrum_en_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy")
                k = np.load(path+f"energy_spectrum_k_{data_name}_{upcale_factor}_hr_{snapshot_num}.npy")
        elif i>0:
            if os.path.exists(path+f"energy_spectrum_k_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy") == False:
                pred = np.load(path+f"{data_name}_{upcale_factor}_{model_name}_pred.npy")[snapshot_num,-1] 
                print(pred.shape)
                en, n = energy_spectrum(2048,2048,pred)
                np.save(path+f"energy_spectrum_en_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy",en)
                k = np.linspace(1,n,n)
                np.save(path+f"energy_spectrum_k_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy",k)
            else:
                en = np.load(path+f"energy_spectrum_en_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy")
                k = np.load(path+f"energy_spectrum_k_{data_name}_{upcale_factor}_{model_name}_{snapshot_num}.npy")
        plt.loglog(k,en[1:]*k**5,label=title_list[i])
    
    plt.legend(fontsize=fontsize,bbox_to_anchor=(0.5, -0.2), loc='lower center',ncol=3)
    plt.xlabel("Wavenumber k",fontsize=fontsize)
    plt.ylabel("Engergy Spectrum E(k)",fontsize=fontsize)
    plt.xticks(fontsize=fontsize-1)
    plt.yticks(fontsize=fontsize-1)
    plt.xlim(1,300)
    # plt.ylim(1e-12,1)
    ax = plt.gca()
    from matplotlib.ticker import LogLocator
    ax.yaxis.set_major_locator(LogLocator(numticks=5))
    ax.xaxis.set_major_locator(LogLocator(numticks=3))
    fig.savefig(f"energy_spectrum_{data_name}_{upcale_factor}_snpt{snapshot_num}.png",dpi=300)
    fig.savefig(f"energy_spectrum_{data_name}_{upcale_factor}_snpt{snapshot_num}.pdf")
    return True

def energy_spectrum(nx,ny,w):
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
    dx = 2.0*np.pi/nx
    dy = 2.0*np.pi/ny
    kx[0:int(nx/2)] = 2*np.pi/(np.float32(nx)*dx)*np.float32(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float32(nx)*dx)*np.float32(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
#        for i in range(1,nx):
#            for j in range(1,ny):          
#                kk1 = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
#                if ( kk1>(k-0.5) and kk1<(k+0.5) ):
#                    ic = ic+1
#                    en[k] = en[k] + es[i,j]
                    
        en[k] = en[k]/ic
        
    return en, n

def energy_spectrum_v2(u,v):
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from math import sqrt
    data = np.stack((u,v),axis=1)
    print ("shape of data = ",data.shape)
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Reading files...localtime",localtime, "- END\n")
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Computing spectrum... ",localtime)
    N = data.shape[-1]
    M= data.shape[-2]
    print("N =",N)
    print("M =",M)
    eps = 1e-16 # to void log(0)
    U = data[:,0].mean(axis=0)
    V = data[:,1].mean(axis=0)
    amplsU = abs(np.fft.fftn(U)/U.size)
    amplsV = abs(np.fft.fftn(V)/V.size)
    print(f"amplsU.shape = {amplsU.shape}")
    EK_U  = amplsU**2
    EK_V  = amplsV**2 
    EK_U = np.fft.fftshift(EK_U)
    EK_V = np.fft.fftshift(EK_V)
    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]
    box_sidex = sign_sizex
    box_sidey = sign_sizey
    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)
    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)
    print ("box sidex     =",box_sidex) 
    print ("box sidey     =",box_sidey) 
    print ("sphere radius =",box_radius )
    print ("centerbox     =",centerx)
    print ("centerboy     =",centery)
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    for i in range(box_sidex):
        for j in range(box_sidey):          
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j]
            EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j]     
    EK_avsphr = 0.5*(EK_U_avsphr + EK_V_avsphr)
    realsize = len(np.fft.rfft(U[:,0]))
    TKEofmean_discrete = 0.5*(np.sum(U/U.size)**2+np.sum(V/V.size)**2)
    TKEofmean_sphere   = EK_avsphr[0]
    total_TKE_discrete = np.sum(0.5*(U**2+V**2))/(N*M) # average over whole domaon / divied by total pixel-value
    total_TKE_sphere   = np.sum(EK_avsphr)
    result_dict = {
    "Real Kmax": realsize,
    "Spherical Kmax": len(EK_avsphr),
    "KE of the mean velocity discrete": TKEofmean_discrete,
    "KE of the mean velocity sphere": TKEofmean_sphere,
    "Mean KE discrete": total_TKE_discrete,
    "Mean KE sphere": total_TKE_sphere
    }
    print(result_dict)
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Computing spectrum... ",localtime, "- END \n")
    return EK_avsphr,realsize

if __name__ == "__main__":
    plot_energy_spectrum_v2(power=0)
    plot_energy_spectrum_v2(power=3)
    plot_energy_spectrum_v2(power=5)
    print("Done")