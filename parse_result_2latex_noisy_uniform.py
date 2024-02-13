import json

placeholder_value = "N/A"

def generate_row(dataset, data, scale):
    row = f"{dataset} & {scale}"
    metrics = ["MSE", "MAE", "RFNE", "IN", "PSNR", "SSIM"]
    for metric in metrics:
        interp_error = data["metrics"][metric]["test1 error"]

        row += f" & {interp_error:.2f}" if isinstance(interp_error, (float, int)) else f" & {placeholder_value}"
    for metric in metrics:
        extrap_error = data["metrics"][metric]["test2 error"]
        row += f" & {extrap_error:.2f}" if isinstance(extrap_error, (float, int)) else f" & {placeholder_value}"
    row += f" & {data['parameters']:.2f} M \\\\"
    return row

def generate_latex_table_from_json(json_file,noise_level):
    with open(json_file, "r") as file:
        data = json.load(file)

    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\caption{Results of SwinIR for \\texttt{SuperBench} datasets with uniform down-sampling and "+str(noise_level) +" noise.}\n"
    latex_table += "\\label{tab:swinir_noise_" +str(noise_level) +"}\n"
    latex_table += "\\centering\n"
    latex_table += "\\scalebox{0.8}{\n"
    latex_table += "\\begin{tabular}{l|c|cccccc|cccccc|c}\n"
    latex_table += "\\toprule\n"
    latex_table += "& UF & \\multicolumn{6}{c}{{Interpolation Errors}} & \\multicolumn{6}{c}{{Extrapolation Errors}} \\\\\n"
    latex_table += "\\cmidrule(r){2-8} \\cmidrule(r){9-14}\n"
    latex_table += "Datasets & ($\\times$) & MSE & MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE & RFNE & IN & PSNR  & SSIM & \\# par.\\\\\n"
    latex_table += "\\midrule\n"

    datasets = ["nskt_16k","nskt_32k","era5","cosmo"]
    scale = 8  # Assuming scale is the same for all datasets

    for dataset in datasets:
        # SwinIR_nskt_16k_noisy_uniform_8_0
        key = f"SwinIR_{dataset}_noisy_uniform_{scale}_{noise_level}"
        if key in data:
            latex_table += generate_row(dataset, data[key], scale) + "\n"
        else:
            latex_table += f"{dataset} & {scale} & " + " & ".join([placeholder_value] * 13) + " \\\\"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}}\n"
    latex_table += "\\end{table}\n"
    return latex_table

# Example usage
latex_table = generate_latex_table_from_json("normed_eval.json",0.1)
print(latex_table)
