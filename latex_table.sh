\begin{table}[h!]
\caption{Results for nskt\_16k\_sim\_4\_v4 dataset with bicubic down-sampling.}
\label{tab:nskt_16k_sim_4_v4_bicubic}
\centering
\scalebox{0.8}{
\begin{tabular}{l|c|cccccc|cccccc|c}
\toprule
& UF & \multicolumn{6}{c}{{Interpolation Errors}} & \multicolumn{6}{c}{{Extrapolation Errors}} \\ 
\cmidrule(r){2-8} \cmidrule(r){8-14} 
Baselines & ($\times$)& MSE& MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE &RFNE & IN & PSNR  & SSIM & \# par.\\ 
\midrule
Bicubic & 4 & 0.1827 & 0.2335 & 0.2551 & 2.8826 & 25.7820 & 0.7889 & 0.1872 & 0.2352 & 0.2587 & 3.4666 & 25.7114 & 0.7847 & 0.0000 M \\
FNO2D & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
SRCNN & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
subpixelCNN & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
EDSR & 4 & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP & INP \\
WDSR & 4 & 0.0773 & 0.1503 & 0.1738 & 2.3274 & 28.6588 & 0.8631 & 0.0899 & 0.1599 & 0.1862 & 2.8606 & 28.2104 & 0.8532 & 1.3514 M \\
SwinIR & 4 & 0.0573 & 0.1261 & 0.1452 & 1.9720 & 30.5057 & 0.8896 & 0.0877 & 0.1550 & 0.1799 & 2.7354 & 28.7368 & 0.8586 & 11.9002 M \\
\bottomrule
\end{tabular}}
\end{table}

