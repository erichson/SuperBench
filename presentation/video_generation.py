import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import ImageGrid
import imageio
import h5py

import cmocean
scale = 16
x = np.load(f"/pscratch/sd/j/junyi012/superbench_v2/eval_buffer/era5_{scale}_lr.npy")
y = np.load(f"/pscratch/sd/j/junyi012/superbench_v2/eval_buffer/era5_{scale}_SwinIR_pred.npy")

img2 = h5py.File(f"/pscratch/sd/j/junyi012/Decay_Turbulence_small/viz/Decay_turb_small_128x128_79.h5", 'r')
data = img2["tasks"]["vorticity"][()][::10]
print(data.shape)
lr_data = data[:,::scale,::scale]
from matplotlib.animation import FuncAnimation
import cmocean
import seaborn as sns
# Function to update the frames
def update(frame_number, x, y, ax):
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(x[frame_number,0], cmap='coolwarm')
    ax[1].imshow(y[frame_number, 0], cmap='coolwarm')
    ax[0].set_axis_off()
    ax[1].set_axis_off()

def update2(frame_number, x, y, ax):
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(x[0], cmap=sns.color_palette('icefire', as_cmap=True))
    ax[1].imshow(y[frame_number],cmap=sns.color_palette('icefire', as_cmap=True))
    ax[0].set_axis_off()
    ax[1].set_axis_off()

def update_DT_lr(frame_number, x, y, ax):
    ax.clear()
    ax.imshow(x[0],cmap=sns.color_palette('icefire', as_cmap=True))
    ax.set_axis_off()

def update_DT_hr(frame_number, x, y, ax):
    ax.clear()
    ax.imshow(y[frame_number],cmap=sns.color_palette('icefire', as_cmap=True))
    ax.set_axis_off()

def update_ERA_lr(frame_number, x, y, ax):
    ax.clear()
    ax.imshow(y[frame_number,0], cmap='coolwarm')
    ax.set_axis_off()

def update_ERA_hr(frame_number, x, y, ax):
    ax.clear()
    ax.imshow(x[frame_number,0], cmap='coolwarm')
    ax.set_axis_off()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 3), constrained_layout=True)

# # Creating animation
# ani = FuncAnimation(fig, update_DT_lr, frames=30, fargs=(lr_data, data, ax))

# # Save the animation as a gif
# ani.save(f'comparison_animation_lr_{scale}.gif', writer='imagemagick', fps=10)
ani = FuncAnimation(fig, update_ERA_hr, frames=200, fargs=(x, y, ax))

# Save the animation as a gif
ani.save(f'comparison_animation_hr_{scale}.gif', writer='imagemagick', fps=5)
plt.close(fig)
