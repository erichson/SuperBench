import json

placeholder_value = "N/A"

def generate_row(dataset, data, scale):
    row = f"{dataset} & {scale}"
    metrics = ["MSE", "MAE", "RFNE", "IN", "PSNR", "SSIM"]
    for metric in metrics:
        interp_error = data["interpolation"][metric]
        extrap_error = data["extrapolation"][metric]
        row += f" & {interp_error:.2f}" if isinstance(interp_error, (float, int)) else f" & {placeholder_value}"
        row += f" & {extrap_error:.2f}" if isinstance(extrap_error, (float, int)) else f" & {placeholder_value}"
    row += f" & {data['parameters']:.2f} M \\\\"
    return row

def generate_latex_table_from_json(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    latex_table = "\\begin{table}[h!]\n"
    latex_table += "\\caption{Results of SwinIR for \\texttt{SuperBench} datasets with uniform down-sampling and $10\\%$ noise.}\n"
    latex_table += "\\label{tab:swinir_noise_10}\n"
    latex_table += "\\centering\n"
    latex_table += "\\scalebox{0.8}{\n"
    latex_table += "\\begin{tabular}{l|c|cccccc|cccccc|c}\n"
    latex_table += "\\toprule\n"
    latex_table += "& UF & \\multicolumn{6}{c}{{Interpolation Errors}} & \\multicolumn{6}{c}{{Extrapolation Errors}} \\\\\n"
    latex_table += "\\cmidrule(r){2-8} \\cmidrule(r){9-14}\n"
    latex_table += "Datasets & ($\\times$) & MSE & MAE & RFNE  & IN & PSNR & SSIM & MSE & MAE & RFNE & IN & PSNR  & SSIM & \\# par.\\\\\n"
    latex_table += "\\midrule\n"

    datasets = ["Fluid flow (16k)", "Fluid flow (32k)", "Cosmology", "Weather"]
    scale = 8  # Assuming scale is the same for all datasets

    for dataset in datasets:
        key = dataset.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if key in data:
            latex_table += generate_row(dataset, data[key], scale) + "\n"
        else:
            latex_table += f"{dataset} & {scale} & " + " & ".join([placeholder_value] * 13) + " \\\\"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}}\n"
    latex_table += "\\end{table}\n"
    return latex_table

# Example usage
latex_table = generate_latex_table_from_json("your_json_file.json")
print(latex_table)
