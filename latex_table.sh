\begin{table}[h!]
\caption{Results for nskt\_32k\_sim\_4\_v7 dataset with bicubic down-sampling.}
\label{tab:nskt_32k_sim_4_v7_bicubic}
\centering
\scalebox{0.8}{
\begin{tabular}{l|c|cccccc|cccccc|c}
\toprule
& UF & \multicolumn{6}{c}{{Interpolation Errors}} & \multicolumn{6}{c}{{Extrapolation Errors}} \\ 
\cmidrule(r){2-8} \cmidrule(r){8-14} 
Baselines & ($\times$)& MSE& MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE &RFNE & IN & PSNR  & SSIM & \# par.\\ 
\midrule
Bicubic & 4 & 0.2808 & 0.2999 & 0.3348 & 3.8039 & 23.7725 & 0.7298 & 0.2895 & 0.3028 & 0.3409 & 3.8406 & 23.1843 & 0.7188 & 0.0000 M \\
FNO2D & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
SRCNN & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
subpixelCNN & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
EDSR & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
WDSR & 4 & 0.1495 & 0.2153 & 0.2541 & 3.3353 & 25.7945 & 0.7901 & 0.1639 & 0.2227 & 0.2671 & 3.7563 & 24.9147 & 0.7772 & 1.3514 M \\
SwinIR & 4 & 0.1457 & 0.2116 & 0.2495 & 3.3234 & 26.0171 & 0.7950 & 0.1687 & 0.2256 & 0.2705 & 3.6729 & 24.8159 & 0.7749 & 11.9002 M \\
\bottomrule
\end{tabular}}
\end{table}

