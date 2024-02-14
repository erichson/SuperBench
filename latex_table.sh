\begin{table}[h!]
\caption{Results for cosmo\_sim\_8 dataset with bicubic down-sampling.}
\label{tab:cosmo_sim_8_bicubic}
\centering
\scalebox{0.6}{
\begin{tabular}{l|c|cccccc|cccccc|c}
\toprule
& UF & \multicolumn{6}{c}{{Interpolation Errors}} & \multicolumn{6}{c}{{Extrapolation Errors}} \\ 
\cmidrule(r){2-8} \cmidrule(r){8-14} 
Baselines & ($\times$)& MSE& MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE &RFNE & IN & PSNR  & SSIM & \# par.\\ 
\midrule
Bicubic & 8 & 1.08 & 0.73 & 1.01 & 8.60 & 21.28 & 0.33 & 1.07 & 0.73 & 1.01 & 8.58 & 21.33 & 0.34 & 0.00 M \\
FNO2D & 8 & 0.86 & 0.67 & 0.92 & 0.94 & 22.03 & 0.28 & 0.85 & 0.67 & 0.92 & 0.94 & 22.05 & 0.28 & 4.75 M \\
FNO2D_patch & 8 & 0.40 & 0.46 & 0.63 & 0.70 & 16.38 & 0.34 & 0.40 & 0.46 & 0.62 & 0.70 & 16.46 & 0.35 & 4.75 M \\
SRCNN & 8 & 0.49 & 0.50 & 0.69 & 9.95 & 24.49 & 0.46 & 0.48 & 0.50 & 0.69 & 10.03 & 24.53 & 0.46 & 0.06 M \\
subpixelCNN & 8 & 0.45 & 0.48 & 0.66 & 9.76 & 24.83 & 0.47 & 0.45 & 0.48 & 0.66 & 9.76 & 24.85 & 0.48 & 0.30 M \\
EDSR & 8 & 0.45 & 0.48 & 0.67 & 9.79 & 24.80 & 0.48 & 0.45 & 0.48 & 0.67 & 9.87 & 24.82 & 0.48 & 1.66 M \\
WDSR & 8 & 0.45 & 0.48 & 0.66 & 9.72 & 24.85 & 0.48 & 0.45 & 0.48 & 0.66 & 9.79 & 24.88 & 0.48 & 1.38 M \\
SwinIR & 8 & 0.44 & 0.48 & 0.66 & 9.73 & 24.91 & 0.48 & 0.44 & 0.48 & 0.66 & 9.83 & 24.93 & 0.48 & 12.05 M \\
\bottomrule
\end{tabular}}
\end{table}

