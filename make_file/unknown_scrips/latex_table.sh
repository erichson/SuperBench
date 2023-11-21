\begin{table}[h!]
\caption{Results for nskt\_16k\_sim dataset with bicubic down-sampling.}
\label{tab:nskt\_16k\_sim_bicubic}
\centering
\scalebox{0.8}{
\begin{tabular}{l|c|cccccc|cccccc|c}
\toprule
& UF & \multicolumn{6}{c}{{Interpolation Errors}} & \multicolumn{6}{c}{{Extrapolation Errors}} \\ 
\cmidrule(r){2-8} \cmidrule(r){8-14} 
Baselines & ($\times$)& MSE& MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE &RFNE & IN & PSNR  & SSIM & \# par.\\ 
\midrule
Bicubic & 4 & 1.0895 & 0.7366 & 1.0519 & 1.2668 & 13.4338 & 0.2784 & 1.1652 & 0.7715 & 1.1047 & 1.1384 & 13.8553 & 0.2594 & 0.0000 M \\
FNO2D & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
SRCNN & 4 & 0.7217 & 0.6123 & 0.8723 & 1.0811 & 14.3186 & 0.2776 & 0.7644 & 0.6320 & 0.9052 & 1.0054 & 14.9547 & 0.2696 & 0.0693 M \\
subpixelCNN & 4 & 0.6954 & 0.5949 & 0.8532 & 1.0453 & 14.6029 & 0.2943 & 0.7379 & 0.6163 & 0.8875 & 0.9827 & 15.1874 & 0.2830 & 0.2588 M \\
EDSR & 4 & 0.3576 & 0.4213 & 0.6098 & 0.8871 & 17.4860 & 0.4877 & 0.8245 & 0.6522 & 0.9299 & 1.0300 & 14.9832 & 0.2438 & 1.5176 M \\
WDSR & 4 & 0.5995 & 0.5411 & 0.7870 & 0.9545 & 15.5884 & 0.3588 & 0.7413 & 0.6125 & 0.8815 & 0.9626 & 15.5087 & 0.2759 & 1.3514 M \\
SwinIR & 4 & 0.6268 & 0.5565 & 0.8044 & 0.9504 & 15.4352 & 0.3282 & 0.6897 & 0.5882 & 0.8525 & 0.9411 & 15.8134 & 0.3057 & 11.9002 M \\
\bottomrule
\end{tabular}}
\end{table}

