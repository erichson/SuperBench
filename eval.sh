### Cosmo LRSIM x8
# python eval.py --data_name cosmo_sim_8 --upscale_factor 8 --model Bicubic --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2;
# python eval_FNO.py --data_name cosmo_sim_8 --upscale_factor 8 --model FNO2D --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_FNO2D_cosmo_sim_8_8_0.001_bicubic_0.0_5544_514.pt --modes 12; 
python eval_FNO.py --data_name cosmo_sim_8 --upscale_factor 8 --model FNO2D_patch --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_FNO2D_cosmo_sim_8_8_0.001_bicubic_0.0_5544_514.pt --modes 12; 
# python eval.py --data_name cosmo_sim_8 --upscale_factor 8 --model SwinIR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_SwinIR_cosmo_sim_8_8_0.0001_bicubic_0.0_5544_8457.pt; 
# python eval.py --data_name cosmo_sim_8 --upscale_factor 8 --model WDSR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_WDSR_cosmo_sim_8_8_0.0001_bicubic_0.0_5544_5323.pt; 
# python eval.py --data_name cosmo_sim_8 --upscale_factor 8 --model EDSR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_EDSR_cosmo_sim_8_8_0.001_bicubic_0.0_5544_5388.pt; 
# python eval.py --data_name cosmo_sim_8 --upscale_factor 8 --model subpixelCNN --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_subpixelCNN_cosmo_sim_8_8_0.001_bicubic_0.0_5544_9538.pt; 
# python eval.py --data_name cosmo_sim_8 --upscale_factor 8 --model SRCNN --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2 --in_channels 2 --model_path results/model_SRCNN_cosmo_sim_8_8_0.001_bicubic_0.0_5544_3032.pt; 

# ### Cosmo BICUBIC x8
# python eval.py --data_name cosmo --upscale_factor 8 --model Bicubic --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2;
# python eval_FNO.py --data_name cosmo --upscale_factor 8 --model FNO2D --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_FNO2D_cosmo_8_0.001_bicubic_0.0_5544_852.pt; 
python eval_FNO.py --data_name cosmo --upscale_factor 8 --model FNO2D_patch --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_FNO2D_cosmo_8_0.001_bicubic_0.0_5544_852.pt --modes 12; 
# python eval.py --data_name cosmo --upscale_factor 8 --model SwinIR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_SwinIR_cosmo_8_0.0001_bicubic_0.0_5544_7689.pt; 
# python eval.py --data_name cosmo --upscale_factor 8 --model WDSR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_WDSR_cosmo_8_0.0001_bicubic_0.0_5544_3487.pt; 
# python eval.py --data_name cosmo --upscale_factor 8 --model EDSR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_EDSR_cosmo_8_0.001_bicubic_0.0_5544_2097.pt; 
# python eval.py --data_name cosmo --upscale_factor 8 --model subpixelCNN --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_subpixelCNN_cosmo_8_0.001_bicubic_0.0_5544_9937.pt; 
# python eval.py --data_name cosmo --upscale_factor 8 --model SRCNN --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_SRCNN_cosmo_8_0.001_bicubic_0.0_5544_1889.pt; 

### Cosmo BICUBIC x16
# python eval.py --data_name cosmo --upscale_factor 16 --model Bicubic --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2;
# python eval_FNO.py --data_name cosmo --upscale_factor 16 --model FNO2D --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_FNO2D_cosmo_16_0.001_bicubic_0.0_5544_50.pt; 
python eval_FNO.py --data_name cosmo --upscale_factor 16 --model FNO2D_patch --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_FNO2D_cosmo_16_0.001_bicubic_0.0_5544_50.pt --modes 12; 
# python eval.py --data_name cosmo --upscale_factor 16 --model SwinIR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_SwinIR_cosmo_16_0.0001_bicubic_0.0_5544_2648.pt; 
# python eval.py --data_name cosmo --upscale_factor 16 --model WDSR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_WDSR_cosmo_16_0.0001_bicubic_0.0_5544_7973.pt; 
# python eval.py --data_name cosmo --upscale_factor 16 --model EDSR --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_EDSR_cosmo_16_0.001_bicubic_0.0_5544_201.pt; 
# python eval.py --data_name cosmo --upscale_factor 16 --model subpixelCNN --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_subpixelCNN_cosmo_16_0.001_bicubic_0.0_5544_6531.pt; 
# python eval.py --data_name cosmo --upscale_factor 16 --model SRCNN --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model_path results/model_SRCNN_cosmo_16_0.001_bicubic_0.0_5544_6451.pt;


### ERA5 BICUBIC x8
python eval_FNO.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/era5_8/model_FNO2D_era5_8_0.001_bicubic_0.0_5544_263.pt --model FNO2D_patch;
# python eval_FNO.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/era5_8/model_FNO2D_era5_8_0.001_bicubic_0.0_5544_263.pt; 
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model WDSR --upscale_factor 8 --model_path results/model_WDSR_era5_8_0.0005_bicubic_0.0_5544_6213.pt
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model EDSR --upscale_factor 8 --model_path results/era5_8/model_EDSR_era5_8_0.0001_bicubic_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/era5_8/model_SwinIR_era5_8_0.0001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model subpixelCNN --upscale_factor 8 --model_path results/era5_8/model_subpixelCNN_era5_8_0.001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SRCNN --upscale_factor 8 --model_path results/era5_8/model_SRCNN_era5_8_0.001_bicubic_0.0_5544_6993.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model Bicubic --upscale_factor 8; 

#### ERA5 BICUBIC x16
python eval_FNO.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/era5_16/model_FNO2D_era5_16_0.001_bicubic_0.0_5544_811.pt --model FNO2D_patch;
# python eval_FNO.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/era5_16/model_FNO2D_era5_16_0.001_bicubic_0.0_5544_811.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model EDSR --upscale_factor 16 --model_path results/era5_16/model_EDSR_era5_16_0.0001_bicubic_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SwinIR --upscale_factor 16 --model_path results/era5_16/model_SwinIR_era5_16_0.0001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model subpixelCNN --upscale_factor 16 --model_path results/era5_16/model_subpixelCNN_era5_16_0.001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SRCNN --upscale_factor 16 --model_path results/era5_16/model_SRCNN_era5_16_0.001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model Bicubic --upscale_factor 16; 
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model WDSR --upscale_factor 16 --model_path results/era5_16/model_WDSR_era5_16_0.0001_bicubic_5544.pt;

### NSKT 16k BICUBIC x8
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model Bicubic --upscale_factor 8; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model WDSR --upscale_factor 8 --model_path results/model_WDSR_nskt_16k_8_0.0002_bicubic_0.0_5544_1303.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model EDSR --upscale_factor 8 --model_path results/model_EDSR_nskt_16k_8_0.001_bicubic_0.0_5544_5321.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SRCNN --upscale_factor 8 --model_path results/model_SRCNN_nskt_16k_8_0.002_bicubic_0.0_5544_6350.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model subpixelCNN --upscale_factor 8 --model_path results/model_subpixelCNN_nskt_16k_8_0.001_bicubic_0.0_5544_5277.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_bicubic_0.0_5544_3094.pt; 
# python eval_FNO.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/model_FNO2D_nskt_16k_8_0.001_bicubic_0.0_5544_927.pt --model FNO2D;
python eval_FNO.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/model_FNO2D_nskt_16k_8_0.001_bicubic_0.0_5544_927.pt --model FNO2D_patch ;

### NSKT 16k BICUBIC x16
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model WDSR --upscale_factor 16 --model_path results/model_WDSR_nskt_16k_16_0.0002_bicubic_0.0_5544_7264.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model EDSR --upscale_factor 16 --model_path results/model_EDSR_nskt_16k_16_0.001_bicubic_0.0_5544_4568.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model Bicubic --upscale_factor 16; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SRCNN --upscale_factor 16 --model_path results/model_SRCNN_nskt_16k_16_0.001_bicubic_0.0_5544_2572.pt;
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model subpixelCNN --upscale_factor 16 --model_path results/model_subpixelCNN_nskt_16k_16_0.001_bicubic_0.0_5544_47.pt; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR --upscale_factor 16 --model_path results/model_SwinIR_nskt_16k_16_0.0001_bicubic_0.0_5544_4150.pt; 
python eval_FNO.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/model_FNO2D_nskt_16k_16_0.001_bicubic_0.0_5544_557.pt --model FNO2D_patch;
# python eval_FNO.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/model_FNO2D_nskt_16k_16_0.001_bicubic_0.0_5544_557.pt --model FNO2D;

### NSKT 32k BICUBIC x8
# python eval_FNO.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/model_FNO2D_nskt_32k_8_0.001_bicubic_0.0_5544_410.pt;
python eval_FNO.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --model FNO2D_patch --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/model_FNO2D_nskt_32k_8_0.001_bicubic_0.0_5544_410.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model Bicubic --upscale_factor 8; 
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model WDSR --upscale_factor 8 --model_path results/model_WDSR_nskt_32k_8_0.0002_bicubic_0.0_5544_786.pt; 
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SRCNN --upscale_factor 8 --model_path results/model_SRCNN_nskt_32k_8_0.001_bicubic_0.0_5544_3390.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model EDSR --upscale_factor 8 --model_path results/model_EDSR_nskt_32k_8_0.001_bicubic_0.0_5544_3668.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model subpixelCNN --upscale_factor 8 --model_path results/model_subpixelCNN_nskt_32k_8_0.001_bicubic_0.0_5544_715.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_nskt_32k_8_0.0001_bicubic_0.0_5544_3173.pt;

### NSKT 32k BICUBIC x16
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model Bicubic --upscale_factor 16; 
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model WDSR --upscale_factor 16 --model_path results/model_WDSR_nskt_32k_16_0.0002_bicubic_0.0_5544_8497.pt; 
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SRCNN --upscale_factor 16 --model_path results/model_SRCNN_nskt_32k_16_0.001_bicubic_0.0_5544_1735.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model EDSR --upscale_factor 16 --model_path results/model_EDSR_nskt_32k_16_0.001_bicubic_0.0_5544_32.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model subpixelCNN --upscale_factor 16 --model_path results/model_subpixelCNN_nskt_32k_16_0.001_bicubic_0.0_5544_618.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SwinIR --upscale_factor 16 --model_path results/model_SwinIR_nskt_32k_16_0.0001_bicubic_0.0_5544_6446.pt;
# python eval_FNO.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/model_FNO2D_nskt_32k_16_0.001_bicubic_0.0_5544_56.pt;
python eval_FNO.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --model FNO2D_patch --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/model_FNO2D_nskt_32k_16_0.001_bicubic_0.0_5544_56.pt;

### NSKT 16k x8 SwinIR + Physics GridSearch
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR_p_1 --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_bicubic_0.0_5544_0.1_7699.pt;
# # python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR_p_01 --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_bicubic_0.0_5544_0.01_643.pt;
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR_p_0001 --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_bicubic_0.0_5544_0.0001_7336.pt;


### NSKT 16k 32k + 0.001 Physics Loss
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR_p_001 --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_bicubic_0.0_5544_0.001_6086.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SwinIR_p_001 --upscale_factor 8 --model_path results/model_SwinIR_nskt_32k_8_0.0001_bicubic_0.0_5544_0.001_9711.pt;
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR_p_001 --upscale_factor 16 --model_path results/model_SwinIR_nskt_16k_16_0.0001_bicubic_0.0_5544_0.001_3552.pt;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SwinIR_p_001 --upscale_factor 16 --model_path results/model_SwinIR_nskt_32k_16_0.0001_bicubic_0.0_5544_0.001_3744.pt;


## NSKT 16k LRSIM x4 x32 patches
# python eval.py --data_name nskt_16k_sim_4_v7 --data_path /pscratch/sd/j/junyi012/nskt/nskt_16k_sim_4_v7 --in_channels 3 --model Bicubic --upscale_factor 4;
# python eval.py --data_name nskt_16k_sim_4_v7 --data_path /pscratch/sd/j/junyi012/nskt/nskt_16k_sim_4_v7 --in_channels 3 --model WDSR --upscale_factor 4 --model_path results/model_WDSR_nskt_16k_sim_4_v4_4_0.0008_bicubic_0.0_5544_488.pt;
# python eval.py --data_name nskt_16k_sim_4_v7 --data_path /pscratch/sd/j/junyi012/nskt/nskt_16k_sim_4_v7 --in_channels 3 --model SwinIR --upscale_factor 4 --model_path results/model_SwinIR_nskt_16k_sim_4_v4_4_0.0008_bicubic_0.0_5544_9572.pt;

# ## NSKT 32k LRSIM x4 x32 patches 
# python eval.py --data_name nskt_32k_sim_4_v7 --data_path /pscratch/sd/j/junyi012/nskt/nskt_32k_sim_4_v7 --in_channels 3 --model Bicubic --upscale_factor 4;
# python eval.py --data_name nskt_32k_sim_4_v7 --data_path /pscratch/sd/j/junyi012/nskt/nskt_32k_sim_4_v7 --in_channels 3 --model WDSR --upscale_factor 4 --model_path results/model_WDSR_nskt_32k_sim_4_v4_4_0.0008_bicubic_0.0_5544_3853.pt;
# python eval.py --data_name nskt_32k_sim_4_v7 --data_path /pscratch/sd/j/junyi012/nskt/nskt_32k_sim_4_v7 --in_channels 3 --model SwinIR --upscale_factor 4 --model_path results/model_SwinIR_nskt_32k_sim_4_v4_4_0.0008_bicubic_0.0_5544_1830.pt;

## NSKT 16k LRSIM x4 x8 patches
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model Bicubic --upscale_factor 4;
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model WDSR --upscale_factor 4 --model_path results/model_WDSR_nskt_16k_sim_4_v7_4_0.0001_bicubic_0.0_5544_3656.pt;
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model SwinIR --upscale_factor 4 --model_path results/model_SwinIR_nskt_16k_sim_4_v7_4_0.0001_bicubic_0.0_5544_2359.pt;
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model EDSR --upscale_factor 4 --model_path results/model_EDSR_nskt_16k_sim_4_v7_4_0.001_bicubic_0.0_5544_5845.pt;
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model SRCNN --upscale_factor 4 --model_path results/model_SRCNN_nskt_16k_sim_4_v7_4_0.001_bicubic_0.0_5544_5653.pt;
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model subpixelCNN --upscale_factor 4 --model_path results/model_subpixelCNN_nskt_16k_sim_4_v7_4_0.001_bicubic_0.0_5544_594.pt;
# python eval_FNO.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --modes 12 --upscale_factor 4 --model_path results/model_FNO2D_nskt_16k_sim_4_v7_4_0.001_bicubic_0.0_5544_719.pt
python eval_FNO.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --modes 12 --upscale_factor 4 --model FNO2D_patch --model_path results/model_FNO2D_nskt_16k_sim_4_v7_4_0.001_bicubic_0.0_5544_719.pt;
# python eval.py --data_name nskt_16k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7 --in_channels 3 --model SwinIR_p001 --upscale_factor 4 --model_path results/model_SwinIR_nskt_16k_sim_4_v7_4_0.0001_bicubic_0.0_5544_0.001_879.pt;

## NSKT 32k LRSIM x4 x8 patches
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model Bicubic --upscale_factor 4;
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model WDSR --upscale_factor 4 --model_path results/model_WDSR_nskt_32k_sim_4_v7_4_0.0001_bicubic_0.0_5544_5766.pt;
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model SwinIR --upscale_factor 4 --model_path results/model_SwinIR_nskt_32k_sim_4_v7_4_0.0001_bicubic_0.0_5544_318.pt;
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model SwinIR_p001 --upscale_factor 4 --model_path results/model_SwinIR_nskt_32k_sim_4_v7_4_0.0001_bicubic_0.0_5544_0.001_9338.pt;
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model EDSR --upscale_factor 4 --model_path results/model_EDSR_nskt_32k_sim_4_v7_4_0.001_bicubic_0.0_5544_6814.pt;
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model SRCNN --upscale_factor 4 --model_path results/model_SRCNN_nskt_32k_sim_4_v7_4_0.001_bicubic_0.0_5544_9857.pt;
# python eval.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --model subpixelCNN --upscale_factor 4 --model_path results/model_subpixelCNN_nskt_32k_sim_4_v7_4_0.001_bicubic_0.0_5544_1847.pt;
# python eval_FNO.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --modes 12 --upscale_factor 4 --model_path results/model_FNO2D_nskt_32k_sim_4_v7_4_0.001_bicubic_0.0_5544_239.pt;
python eval_FNO.py --data_name nskt_32k_sim_4_v8 --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7 --in_channels 3 --modes 12 --upscale_factor 4 --model FNO2D_patch --model_path results/model_FNO2D_nskt_32k_sim_4_v7_4_0.001_bicubic_0.0_5544_239.pt;

### SwinIR + Noise 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_noisy_uniform_0.05_5544_9073.pt --noise_ratio 0.05 --method noisy_uniform;
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_nskt_16k_8_0.0001_noisy_uniform_0.1_5544_4754.pt --noise_ratio 0.1 --method noisy_uniform;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_nskt_32k_8_0.0001_noisy_uniform_0.05_5544_8822.pt --noise_ratio 0.05 --method noisy_uniform;
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_nskt_32k_8_0.0001_noisy_uniform_0.1_5544_756.pt --noise_ratio 0.1 --method noisy_uniform;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_new_era5_8_0.0001_noisy_uniform_0.05_5544.pt --noise_ratio 0.05 --method noisy_uniform;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_new_era5_8_0.0001_noisy_uniform_0.1_5544.pt --noise_ratio 0.1 --method noisy_uniform;
# python eval.py --data_name cosmo --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_cosmo_8_0.0001_noisy_uniform_0.05_5544_2035.pt --noise_ratio 0.05 --method noisy_uniform;
# python eval.py --data_name cosmo --data_path /pscratch/sd/j/junyi012/superbench_v2/cosmo_v2 --in_channels 2 --model SwinIR --upscale_factor 8 --model_path results/model_SwinIR_cosmo_8_0.0001_noisy_uniform_0.1_5544_2273.pt --noise_ratio 0.1 --method noisy_uniform;
