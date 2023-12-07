\begin{table}[h!]
\caption{Results for nskt\_32k\_sim\_4\_v4 dataset with bicubic down-sampling.}
\label{tab:nskt\_32k\_sim\_4\_v4_bicubic}
\centering
\scalebox{0.8}{
\begin{tabular}{l|c|cccccc|cccccc|c}
\toprule
& UF & \multicolumn{6}{c}{{Interpolation Errors}} & \multicolumn{6}{c}{{Extrapolation Errors}} \\ 
\cmidrule(r){2-8} \cmidrule(r){8-14} 
Baselines & ($\times$)& MSE& MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE &RFNE & IN & PSNR  & SSIM & \# par.\\ 
\midrule
Bicubic & 4 & 0.0914 & 0.1557 & 0.1766 & 2.2641 & 29.0568 & 0.8679 & 0.1013 & 0.1624 & 0.1841 & 2.1766 & 29.1250 & 0.8600 & 0.0000 M \\
FNO2D & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
SRCNN & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
subpixelCNN & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
EDSR & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
WDSR & 4 & 0.0399 & 0.1091 & 0.1251 & 1.7173 & 31.5672 & 0.9112 & 0.0485 & 0.1162 & 0.1353 & 1.9735 & 31.3289 & 0.9036 & 1.3514 M \\
SwinIR & 4 & 0.0360 & 0.1080 & 0.1245 & 1.6297 & 31.3245 & 0.9128 & 0.0523 & 0.1227 & 0.1436 & 2.0225 & 30.6435 & 0.8973 & 11.9002 M \\
\bottomrule
\end{tabular}}
\end{table}

