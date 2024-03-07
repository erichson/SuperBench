import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
import pyfftw
import cmocean

def plot_energy_spectrum(
    data_name = "nskt_32k",
    upcale_factor=16, 
    snapshot_num=16,
    cmap=cmocean.cm.balance):
    if data_name !='era5':
        model_saved_list = ['Bicubic', 'SRCNN', 'FNO2D_patch','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['HR','Bicubic', 'SRCNN','FNO$^*$', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]
    else:
        model_saved_list = ['Bicubic', 'SRCNN', 'FNO2D','subpixelCNN', 'EDSR', 'WDSR','SwinIR', ]
        title_list = ['HR','Bicubic', 'SRCNN','FNO', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR',]


    # get data and data range
    path = "/pscratch/sd/j/junyi012/superbench_v2/eval_buffer/"
    lr = np.load(path+data_name+"_"+str(upcale_factor)+"_lr.npy")
    hr = np.load(path+data_name+"_"+str(upcale_factor)+"_hr.npy")
    lr = lr[snapshot_num,-1]
    hr = hr[snapshot_num,-1]


    pred_list = [hr]+[np.load(path+f"{data_name}_{model_name}_{upcale_factor}_pred_b{snapshot_num}c{-1}.npy")[0,-1] for model_name in model_saved_list]

    print(len(pred_list))
    print(lr.shape, hr.shape,pred_list[0].shape,pred_list[1].shape,pred_list[2].shape)
    fig = plt.figure(figsize=(5,5))
    for i in range(len(pred_list)):
        en, n = energy_spectrum(2048,2048,pred_list[i])
        k = np.linspace(1,n,n)
        plt.loglog(k,en[1:],label=title_list[i])
    plt.legend()
    fig.savefig(f"energy_spectrum_{data_name}_{upcale_factor}_{snapshot_num}.png",dpi=300)
    fig.savefig(f"energy_spectrum_{data_name}_{upcale_factor}_{snapshot_num}.pdf")
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

if __name__ == "__main__":
    plot_energy_spectrum()
    print("Done")