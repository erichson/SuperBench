# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model EDSR --upscale_factor 8 --model_path results/era5_8/model_EDSR_era5_8_0.0001_bicubic_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SwinIR --upscale_factor 8 --model_path results/era5_8/model_SwinIR_era5_8_0.0001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model subpixelCNN --upscale_factor 8 --model_path results/era5_8/model_subpixelCNN_era5_8_0.001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SRCNN --upscale_factor 8 --model_path results/era5_8/model_SRCNN_era5_8_0.001_bicubic_0.0_5544_6993.pt;

# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model EDSR --upscale_factor 16 --model_path results/era5_16/model_EDSR_era5_16_0.0001_bicubic_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SwinIR --upscale_factor 16 --model_path results/era5_16/model_SwinIR_era5_16_0.0001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model subpixelCNN --upscale_factor 16 --model_path results/era5_16/model_subpixelCNN_era5_16_0.001_bicubic_0.0_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model SRCNN --upscale_factor 16 --model_path results/era5_16/model_SRCNN_era5_16_0.001_bicubic_0.0_5544.pt;

# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model Bicubic --upscale_factor 16; 
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model Bicubic --upscale_factor 8; 
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model WDSR --upscale_factor 16 --model_path results/era5_16/model_WDSR_era5_16_0.0001_bicubic_5544.pt;
# python eval.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --model WDSR --upscale_factor 8 --model_path results/model_WDSR_era5_8_0.0005_bicubic_0.0_5544_6213.pt;

# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model Bicubic --upscale_factor 8; 
# python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model Bicubic --upscale_factor 16; 
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model Bicubic --upscale_factor 8; 
# python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model Bicubic --upscale_factor 16; 
# python eval.py --data_name nskt_32k_sim --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4 --in_channels 3 --model Bicubic --upscale_factor 4; 
# python eval.py --data_name nskt_16k_sim --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4 --in_channels 3 --model Bicubic --upscale_factor 4; 
# python eval_FNO.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --modes 12 --upscale_factor 8 --model_path results/era5_8/model_FNO2D_era5_8_0.001_bicubic_0.0_5544_263.pt; 
# python eval_FNO.py --data_name era5 --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/era5_16/model_FNO2D_era5_16_0.001_bicubic_0.0_5544_811.pt;
python eval_FNO.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --modes 12 --upscale_factor 16 --model_path results/model_FNO2D_nskt_16k_16_0.001_bicubic_0.0_5544_557.pt;
python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model WDSR --upscale_factor 8 --model_path results/model_WDSR_nskt_16k_8_0.0002_bicubic_0.0_5544_1303.pt; 
python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model WDSR --upscale_factor 16 --model_path results/model_WDSR_nskt_16k_16_0.0002_bicubic_0.0_5544_7264.pt; 
python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model WDSR --upscale_factor 8 --model_path results/model_WDSR_nskt_32k_8_0.0002_bicubic_0.0_5544_786.pt; 
python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --in_channels 3 --model WDSR --upscale_factor 16 --model_path results/model_WDSR_nskt_32k_16_0.0002_bicubic_0.0_5544_8497.pt; 
python eval.py --data_name nskt_32k_sim --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4 --in_channels 3 --model WDSR --upscale_factor 4 --model_path results/model_WDSR_nskt_32k_sim_4_0.0005_bicubic_0.0_5544_4747.pt; 
python eval.py --data_name nskt_16k_sim --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4 --in_channels 3 --model WDSR --upscale_factor 4 --model_path results/model_WDSR_nskt_32k_sim_4_0.0005_bicubic_0.0_5544_3549.pt; 
python eval.py --data_name nskt_16k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SRCNN --upscale_factor 8 --model_path results/model_SRCNN_nskt_16k_8_0.002_bicubic_0.0_5544_6350.pt; 
python eval.py --data_name nskt_32k --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --in_channels 3 --model SRCNN --upscale_factor 16 --model_path results/model_SRCNN_nskt_32k_16_0.001_bicubic_0.0_5544_1735.pt;